use image::{Rgba, RgbaImage};
use nalgebra::{DMatrix, DMatrixView, Matrix3, Vector2};

use crate::kernels::{self, EmbossKind};

#[derive(Clone)]
pub struct Operator {
    r: DMatrix<u8>,
    g: DMatrix<u8>,
    b: DMatrix<u8>,
}

impl Operator {
    pub fn from_rgba(image: &RgbaImage) -> Self {
        let (width, height) = image.dimensions();
        let mut r = DMatrix::zeros(height as usize, width as usize);
        let mut g = DMatrix::zeros(height as usize, width as usize);
        let mut b = DMatrix::zeros(height as usize, width as usize);

        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x, y);
                r[(y as usize, x as usize)] = pixel[0];
                g[(y as usize, x as usize)] = pixel[1];
                b[(y as usize, x as usize)] = pixel[2];
            }
        }

        Self { r, g, b }
    }

    pub fn to_image(&self) -> RgbaImage {
        let red = &self.r;
        let green = &self.g;
        let blue = &self.b;

        let (height, width) = (red.nrows() as u32, red.ncols() as u32);
        let mut img = RgbaImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                img.put_pixel(
                    x,
                    y,
                    Rgba([
                        red[(y as usize, x as usize)],
                        green[(y as usize, x as usize)],
                        blue[(y as usize, x as usize)],
                        255,
                    ]),
                );
            }
        }
        img
    }

    #[must_use]
    pub fn box_blur(&self, radius: usize) -> Operator {
        let kernel = kernels::box_kernel(radius);

        let r = box_blur(&self.r, &kernel);
        let b = box_blur(&self.b, &kernel);
        let g = box_blur(&self.g, &kernel);

        Self { r, g, b }
    }

    #[must_use]
    pub fn gaussian_blur(&self, radius: usize) -> Operator {
        let kernel = kernels::gaussian(radius);

        let r = gaussian_blur(&self.r, &kernel);
        let b = gaussian_blur(&self.b, &kernel);
        let g = gaussian_blur(&self.g, &kernel);

        Self { r, g, b }
    }

    #[must_use]
    pub fn sobel_blur(&self) -> Operator {
        let r = sobel_blur(&self.r);
        let b = sobel_blur(&self.b);
        let g = sobel_blur(&self.g);

        Self { r, g, b }
    }

    pub fn emboss(&self, kind: EmbossKind) -> Operator {
        let kernel = kernels::emboss(kind);

        let r = emboss(&self.r, kernel.as_view());
        let b = emboss(&self.b, kernel.as_view());
        let g = emboss(&self.g, kernel.as_view());

        Self { r, g, b }
    }
}

pub fn box_blur(matrix: &DMatrix<u8>, kernel: &DMatrix<f32>) -> DMatrix<u8> {
    apply_convolution(matrix, kernel.as_view(), identity_norm())
}

fn emboss(matrix: &DMatrix<u8>, kernel: DMatrixView<f32>) -> DMatrix<u8> {
    apply_convolution(matrix, kernel, |p| p + 128.0)
}

pub fn gaussian_blur(
    matrix: &DMatrix<u8>,
    kernel: &DMatrix<f32>,
) -> DMatrix<u8> {
    apply_convolution(matrix, kernel.as_view(), kernel_sum(kernel.as_view()))
}

pub fn sobel_blur(matrix: &DMatrix<u8>) -> DMatrix<u8> {
    let sobel_x_kernel = Matrix3::from_row_slice(&[
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0,
    ]);
    // Matrix3::new(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
    let sobel_y_kernel = Matrix3::from_row_slice(&[
        1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0,
    ]);
    // Matrix3::new(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);

    let (height, width) = (matrix.nrows(), matrix.ncols());

    let mut result = DMatrix::zeros(height, width);

    for y in 0..height {
        for x in 0..width {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;

            for i in -1..=1 {
                for j in -1..=1 {
                    let y1 = (i + y as isize).clamp(0, (height - 1) as isize)
                        as usize;
                    let x1 = (j + x as isize).clamp(0, (width - 1) as isize)
                        as usize;

                    let pixel = matrix[(y1, x1)] as f32;
                    sum_x += pixel
                        * sobel_x_kernel[((i + 1) as usize, (j + 1) as usize)];
                    sum_y += pixel
                        * sobel_y_kernel[((i + 1) as usize, (j + 1) as usize)];
                }
            }

            let gradient = Vector2::new(sum_x, sum_y).norm();
            // let gradient = (sum_x.powi(2) + sum_y.powi(2)).sqrt();

            result[(y, x)] = gradient.clamp(0.0, 255.0) as u8;
        }
    }

    result
}

fn apply_convolution<F: Fn(f32) -> f32>(
    matrix: &DMatrix<u8>,
    kernel: DMatrixView<f32>,
    norm: F,
) -> DMatrix<u8> {
    assert!(kernel.is_square());

    let (height, width) = (matrix.nrows(), matrix.ncols());

    let kernel_size = kernel.nrows();
    let kernel_radius = kernel_size / 2;

    let mut result = DMatrix::zeros(height, width);

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            // Apply the kernel to the current pixel
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let y_index = (y as isize
                        + (ky as isize - kernel_radius as isize))
                        .clamp(0, (height - 1) as isize)
                        as usize;
                    let x_index = (x as isize
                        + (kx as isize - kernel_radius as isize))
                        .clamp(0, (width - 1) as isize)
                        as usize;

                    let pixel = matrix[(y_index, x_index)] as f32;
                    sum += pixel * kernel[(ky, kx)];
                }
            }

            result[(y, x)] = norm(sum).clamp(0.0, 255.0) as u8;
        }
    }

    result
}

fn kernel_sum(kernel: DMatrixView<f32>) -> impl Fn(f32) -> f32 {
    let sum = kernel.sum();
    move |p| p / sum
}

fn identity_norm() -> impl Fn(f32) -> f32 {
    |p| p
}
