// TODO: Encode multi-point color gradient as fragment format
//       * Each fragment essentially runs a gradient animation
//       * Animation speed equivalent to existing decay parameter
//       * Allows for fine-grained control over intensity, decay rate, multi-step color
//
// TODO: 3D rendering
//       * Need to figure out how to make a phosphor decay model work with a rotateable camera
//       * Want to avoid UE4-style whole screen smearing
//       * Cubemap-style setup seems like the best approach currently
//         * Is a geometry-based solution to this viable?
//         * Geometry layer using polar coordinates
//         * Tesselate lines to counteract linear transform artifacts
//         * Current phosphor rendering relies on rasterization
//           * Could supersample, or render cubemap texels via shaped point sprites as a design workaround
//         * Alternately, devise an equivalent-looking effect using geometry animation
//           * Viable - current effect could be achieved with fine enough geo
//           * Can take advantage of MSAA to avoid framebuffer size issues
//
//       * MechWarrior 2 gradient skybox background
//         * Setting for underlay / overlay behavior
//         * Overlay acts like a vectrex color overlay
//         * Underlay respects depth and doesn't draw behind solid objects
//
//       * Depth buffer
//         * Draw depth using model triangles if in underlay mode
//         * Draw visuals using lines

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::time::{Duration, Instant};

use euc::{
    buffer::Buffer2d,
    rasterizer::{BackfaceCullingDisabled, Lines, Triangles},
    Interpolate, Pipeline, Target,
};
use minifb::{Key, Window, WindowOptions};

type Position = (f32, f32);

#[derive(Debug, Default, Copy, Clone)]
struct Color(f32, f32, f32);

#[derive(Debug, Default, Clone)]
struct Gradient(Vec<(f32, Color)>);

impl FromIterator<(f32, Color)> for Gradient {
    fn from_iter<T: IntoIterator<Item = (f32, Color)>>(iter: T) -> Self {
        Gradient(iter.into_iter().collect())
    }
}

impl Gradient {
    pub fn black() -> Self {
        [(0.0, BLACK)].into_iter().collect()
    }

    pub fn white() -> Self {
        [(0.0, WHITE)].into_iter().collect()
    }

    pub fn red() -> Self {
        [(0.0, BLACK), (1.0, RED), (1.0, WHITE)]
            .into_iter()
            .collect()
    }

    pub fn green() -> Self {
        [(0.0, BLACK), (1.0, GREEN), (1.0, WHITE)]
            .into_iter()
            .collect()
    }

    pub fn blue() -> Self {
        [(0.0, BLACK), (1.0, BLUE), (1.0, WHITE)]
            .into_iter()
            .collect()
    }

    pub fn delayed(mut grad: Gradient, amount: f32) -> Gradient {
        let mut last = grad.0.pop().unwrap();
        last.0 -= amount;
        Gradient(
            [grad.0, vec![last, (amount, BLACK)]]
                .into_iter()
                .flatten()
                .collect(),
        )
    }

    pub fn sample(&self, f: f32) -> Color {
        let (weights, colors): (Vec<f32>, Vec<Color>) = self.0.iter().copied().unzip();

        let weights_seq: Vec<f32> = (0..weights.len())
            .map(|i| weights.iter().take(i + 1).sum())
            .collect();

        if let Some(idx) = weights_seq.iter().copied().position(|w| w == f) {
            return colors[idx];
        };

        let from = weights_seq.iter().copied().rposition(|w| w < f).unwrap();

        let to = if let Some(to) = weights_seq.iter().copied().position(|w| w > f) {
            to
        } else {
            return colors[from];
        };

        let weight_from = weights_seq[from];
        let weight_to = weights_seq[to];

        let weight_len = weight_to - weight_from;

        let f_local = (f - weight_from) / weight_len;

        Interpolate::lerp2(colors[from], colors[to], 1.0 - f_local, f_local)
    }
}

impl Interpolate for Gradient {
    fn lerp2(a: Self, b: Self, x: f32, y: f32) -> Self {
        let Gradient(a) = a;
        let Gradient(b) = b;

        Gradient(
            a.into_iter()
                .zip(b.into_iter())
                .map(|((ac, aw), (bc, bw))| {
                    (
                        Interpolate::lerp2(ac, bc, x, y),
                        Interpolate::lerp2(aw, bw, x, y),
                    )
                })
                .collect(),
        )
    }

    fn lerp3(a: Self, b: Self, c: Self, x: f32, y: f32, z: f32) -> Self {
        let Gradient(a) = a;
        let Gradient(b) = b;
        let Gradient(c) = c;

        Gradient(
            a.into_iter()
                .zip(b.into_iter().zip(c.into_iter()))
                .map(|((ac, aw), ((bc, bw), (cc, cw)))| {
                    (
                        Interpolate::lerp3(ac, bc, cc, x, y, z),
                        Interpolate::lerp3(aw, bw, cw, x, y, z),
                    )
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient() {
        let grad = Gradient::green();

        println!("Sample left");
        let samp_left = grad.sample(0.0);

        println!("Sample mid left");
        let samp_mid_left = grad.sample(0.5);

        println!("Sample center");
        let samp_center = grad.sample(1.0);

        println!("Sample mid right");
        let samp_mid_right = grad.sample(2.0);

        println!("Sample right");
        let samp_right = grad.sample(3.0);

        println!(
            "{:?} -> {:?} -> {:?} -> {:?} -> {:?}",
            samp_left, samp_mid_left, samp_center, samp_mid_right, samp_right
        );
    }
}

struct Vertex {
    position: Position,
    color: Gradient,
    intensity: f32,
    decay: f32,
}

#[derive(Clone)]
struct Fragment {
    color: Gradient,
    intensity: f32,
    decay: f32,
}

impl std::ops::Add<Self> for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Color(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl std::ops::Sub<Self> for Color {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Color(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl std::ops::Mul<Self> for Color {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Color(self.0 * rhs.0, self.1 * rhs.1, self.2 * rhs.2)
    }
}

impl std::ops::Div<Self> for Color {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Color(self.0 / rhs.0, self.1 / rhs.1, self.2 / rhs.2)
    }
}

impl Interpolate for Color {
    fn lerp2(a: Self, b: Self, x: f32, y: f32) -> Self {
        let Color(ar, ag, ab) = a;
        let Color(br, bg, bb) = b;

        Color(
            Interpolate::lerp2(ar, br, x, y),
            Interpolate::lerp2(ag, bg, x, y),
            Interpolate::lerp2(ab, bb, x, y),
        )
    }

    fn lerp3(a: Self, b: Self, c: Self, x: f32, y: f32, z: f32) -> Self {
        let Color(ar, ag, ab) = a;
        let Color(br, bg, bb) = b;
        let Color(cr, cg, cb) = c;

        Color(
            Interpolate::lerp3(ar, br, cr, x, y, z),
            Interpolate::lerp3(ag, bg, cg, x, y, z),
            Interpolate::lerp3(ab, bb, cb, x, y, z),
        )
    }
}

fn vertex(position: Position, color: Gradient, intensity: f32, decay: f32) -> Vertex {
    Vertex {
        position,
        color,
        intensity,
        decay,
    }
}

struct Example {
    time: f32,
    delta: f32,
}

impl Pipeline for Example {
    type Vertex = Vertex;
    type VsOut = (Gradient, f32, f32);
    type Pixel = Fragment;

    // Vertex shader
    fn vert(&self, vertex: &Self::Vertex) -> ([f32; 4], Self::VsOut) {
        let (x, y) = vertex.position;
        (
            [x * 0.5, y * 0.5, 0.0, 1.0],
            (vertex.color.clone(), vertex.intensity, vertex.decay),
        )
    }

    // Fragment shader
    fn frag(&self, (color, intensity, decay): &Self::VsOut) -> Self::Pixel {
        let intensity = *intensity;
        let decay = *decay;
        Fragment {
            color: color.clone(),
            intensity,
            decay,
        }
    }
}

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

// Reference time metrics
// Phosphor decay parameters
const DECAY_RATE: f32 = 30.0;

const BLACK: Color = Color(0.0, 0.0, 0.0);
const WHITE: Color = Color(1.0, 1.0, 1.0);
const RED: Color = Color(1.0, 0.0, 0.0);
const GREEN: Color = Color(0.0, 1.0, 0.0);
const BLUE: Color = Color(0.0, 0.0, 1.0);

/// Convert a 4-part RGBA8 value to a 1-part RGB32 value
fn to_rgb32(
    Fragment {
        color, intensity, ..
    }: Fragment,
) -> u32 {
    let Color(r, g, b) = color.sample(intensity);

    let mut o = 0u32;
    o |= (r * 255.0).round().clamp(0.0, 255.0) as u32;
    o <<= 8;
    o |= (g * 255.0).round().clamp(0.0, 255.0) as u32;
    o <<= 8;
    o |= (b * 255.0).round().clamp(0.0, 255.0) as u32;
    o
}

fn phosphor_decay(
    Fragment {
        color,
        intensity,
        decay,
    }: &mut Fragment,
    delta: f32,
) {
    *intensity = (*intensity - *decay * delta).max(0.0);
}

struct Point {
    pos: Position,
    prev_pos: Position,
    color: Gradient,
    intensity: f32,
    decay: f32,
}

impl Point {
    fn new(pos: Position, color: Gradient, intensity: f32, decay: f32) -> Self {
        Point {
            pos,
            prev_pos: pos,
            color,
            intensity,
            decay,
        }
    }

    fn set_pos(&mut self, pos: (f32, f32)) {
        self.prev_pos = self.pos;
        self.pos = pos;
    }

    fn draw(
        &self,
        example: &mut Example,
        color: &mut Buffer2d<Fragment>,
        depth: &mut Buffer2d<f32>,
    ) {
        example.draw::<Lines<_>, _>(
            &[
                vertex(
                    (self.prev_pos.0, self.prev_pos.1),
                    self.color.clone(),
                    self.intensity,
                    self.decay,
                ),
                vertex(
                    (self.pos.0, self.pos.1),
                    self.color.clone(),
                    self.intensity,
                    self.decay,
                ),
            ],
            color,
            Some(depth),
        );
    }
}

struct Line {
    v0: Point,
    v1: Point,
}

impl Line {
    fn new(v0: Point, v1: Point) -> Self {
        Line { v0, v1 }
    }

    fn set_v0(&mut self, v0: (f32, f32)) {
        self.v0.set_pos(v0);
    }

    fn set_v1(&mut self, v1: (f32, f32)) {
        self.v1.set_pos(v1);
    }

    fn draw(
        &self,
        example: &mut Example,
        color: &mut Buffer2d<Fragment>,
        depth: &mut Buffer2d<f32>,
    ) {
        example.draw::<Triangles<_, BackfaceCullingDisabled>, _>(
            &[
                vertex(
                    self.v0.pos,
                    self.v0.color.clone(),
                    self.v0.intensity,
                    self.v0.decay,
                ),
                vertex(
                    self.v1.pos,
                    self.v1.color.clone(),
                    self.v1.intensity,
                    self.v1.decay,
                ),
                vertex(
                    self.v0.prev_pos,
                    self.v0.color.clone(),
                    self.v0.intensity,
                    self.v0.decay,
                ),
                vertex(
                    self.v1.pos,
                    self.v1.color.clone(),
                    self.v1.intensity,
                    self.v1.decay,
                ),
                vertex(
                    self.v1.prev_pos,
                    self.v1.color.clone(),
                    self.v1.intensity,
                    self.v1.decay,
                ),
                vertex(
                    self.v0.prev_pos,
                    self.v0.color.clone(),
                    self.v0.intensity,
                    self.v0.decay,
                ),
            ],
            color,
            Some(depth),
        );
    }
}

fn main() {
    let mut color = Buffer2d::new(
        [WIDTH, HEIGHT],
        Fragment {
            color: Gradient::black(),
            intensity: 0.0,
            decay: DECAY_RATE,
        },
    );
    let mut depth = Buffer2d::new([WIDTH, HEIGHT], 1.0);

    let mut window = Window::new(
        "Software Rendering",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap();

    let scan_rate = Duration::from_secs_f32(0.5);

    let mut example = Example {
        time: 0.0,
        delta: 0.0,
    };
    let mut ts = Instant::now();
    let mut scan_ts = Instant::now();
    let start = ts;

    let mut point0 = Point::new((0.0, 0.0), Gradient::green(), 4.0, 30.0);
    let mut point1 = Point::new((0.0, 0.0), Gradient::green(), 4.0, 60.0);
    let mut point2 = Point::new((0.0, 0.0), Gradient::green(), 4.0, 120.0);

    let mut line0 = Line::new(
        Point::new((0.0, 0.0), Gradient::red(), 4.0, 30.0),
        Point::new((0.0, 0.0), Gradient::red(), 4.0, 30.0),
    );
    let mut line1 = Line::new(
        Point::new((0.0, 0.0), Gradient::red(), 4.0, 60.0),
        Point::new((0.0, 0.0), Gradient::red(), 4.0, 60.0),
    );
    let mut line2 = Line::new(
        Point::new((0.0, 0.0), Gradient::red(), 4.0, 120.0),
        Point::new((0.0, 0.0), Gradient::red(), 4.0, 120.0),
    );

    let mut fixed_tick = true;
    let mut multisample = true;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = if fixed_tick {
            ts + Duration::from_millis(16)
        } else {
            Instant::now()
        };

        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            multisample = !multisample;
        }

        if window.is_key_pressed(Key::Backspace, minifb::KeyRepeat::No) {
            fixed_tick = !fixed_tick;
        }

        let time = now.duration_since(start);
        let delta = now.duration_since(ts);

        let time_secs = time.as_secs_f32();
        let delta_secs = delta.as_secs_f32();

        color
            .as_mut()
            .iter_mut()
            .for_each(|p| phosphor_decay(p, delta_secs));

        example.time = time_secs;
        example.delta = delta_secs;

        // Clear depth buffer
        depth.as_mut().fill(1.0);

        // Draw reference line
        example.draw::<Lines<_>, _>(
            &[
                vertex((-1.5, -1.5), Gradient::green(), 0.0, 30.0),
                vertex((1.5, -1.5), Gradient::green(), 2.0, 30.0),
            ],
            &mut color,
            Some(&mut depth),
        );

        let timer_0 = time_secs * 3.33;
        let timer_1 = timer_0 * 4.0;

        // Draw moving point
        let point_move_sin = timer_0.sin() * 0.2;
        let point_move_cos = timer_1.sin() * 0.2;

        point0.set_pos((-1.5 + point_move_sin, 1.5 + point_move_cos));
        point0.draw(&mut example, &mut color, &mut depth);

        point1.set_pos((-1.0 + point_move_sin, 1.5 + point_move_cos));
        point1.draw(&mut example, &mut color, &mut depth);

        point2.set_pos((-0.5 + point_move_sin, 1.5 + point_move_cos));
        point2.draw(&mut example, &mut color, &mut depth);

        line0.set_v0((-1.4 + point_move_sin, 0.0 + point_move_cos));
        line0.set_v1((-1.6 + point_move_sin, 0.0 + point_move_cos));
        line0.draw(&mut example, &mut color, &mut depth);

        line1.set_v0((-0.9 + point_move_sin, 0.0 + point_move_cos));
        line1.set_v1((-1.1 + point_move_sin, 0.0 + point_move_cos));
        line1.draw(&mut example, &mut color, &mut depth);

        line2.set_v0((-0.4 + point_move_sin, 0.0 + point_move_cos));
        line2.set_v1((-0.6 + point_move_sin, 0.0 + point_move_cos));
        line2.draw(&mut example, &mut color, &mut depth);

        // Scan triangle if timer has expired
        if now.duration_since(scan_ts) > scan_rate {
            example.draw::<Lines<_>, _>(
                &[
                    vertex(
                        (-1.0, -1.0),
                        Gradient::delayed(Gradient::red(), 0.05),
                        2.0,
                        30.0,
                    ),
                    vertex(
                        (1.0, -1.0),
                        Gradient::delayed(Gradient::red(), 0.05),
                        2.0,
                        30.0,
                    ),
                    vertex(
                        (1.0, -1.0),
                        Gradient::delayed(Gradient::green(), 0.05),
                        3.0,
                        30.0,
                    ),
                    vertex(
                        (0.0, 1.0),
                        Gradient::delayed(Gradient::green(), 0.05),
                        4.0,
                        30.0,
                    ),
                    vertex(
                        (0.0, 1.0),
                        Gradient::delayed(Gradient::blue(), 0.05),
                        4.0,
                        30.0,
                    ),
                    vertex(
                        (-1.0, -1.0),
                        Gradient::delayed(Gradient::blue(), 0.05),
                        5.0,
                        30.0,
                    ),
                ],
                &mut color,
                Some(&mut depth),
            );

            scan_ts = now;
        }

        let samples_per_axis = 2isize;

        let buf = (0..HEIGHT)
            .into_iter()
            .flat_map(|y| (0..WIDTH).into_iter().map(move |x| (x, y)))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(x, y)| {
                let mut p = unsafe { color.get([x, y]) };

                /*
                if multisample {
                    for iy in -samples_per_axis..samples_per_axis {
                        for ix in -samples_per_axis..samples_per_axis {
                            if x > samples_per_axis as usize
                                && x < WIDTH - 1 - samples_per_axis as usize
                                && y > samples_per_axis as usize
                                && y < HEIGHT - 1 - samples_per_axis as usize
                            {
                                let fx = ix.abs() as f32;
                                let fy = iy.abs() as f32;

                                let weight =
                                    (fx.powi(2) + fy.powi(2)).sqrt() / samples_per_axis as f32;
                                let weight = (1.0 - weight).clamp(0.0, 1.0);
                                let weight = 1.0 - (1.0 - weight.powi(2)).sqrt();

                                let x = (x as isize + ix) as usize;
                                let y = (y as isize + iy) as usize;

                                let c = unsafe { color.get([x, y]) };
                                p.color = p.color + c.color * Color(weight, weight, weight);
                            }
                        }
                    }
                }
                */

                p
            })
            .collect::<Vec<Fragment>>();

        let buf = buf.into_iter().map(to_rgb32).collect::<Vec<_>>();
        window.update_with_buffer(&buf, WIDTH, HEIGHT).unwrap();

        ts = now;

        println!("Delta: {}", delta_secs);
    }
}
