use image::{Rgba, RgbaImage};
use nalgebra::{DMatrix, DMatrixView, Matrix3};

use crate::kernels;

#[derive(Clone)]
pub struct Operator {
    r: DMatrix<u8>,
    g: DMatrix<u8>,
    b: DMatrix<u8>,
}

impl Operator {
    pub fn from_image(image: &RgbaImage) -> Self {
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
    pub fn box_blur(&self) -> Operator {
        let r = box_blur(&self.r);
        let b = box_blur(&self.b);
        let g = box_blur(&self.g);

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
}

pub fn box_blur(matrix: &DMatrix<u8>) -> DMatrix<u8> {
    let kernel = Matrix3::from_element(1.0 / 9.0);

    apply_convolution(matrix, kernel.as_view())
}

pub fn gaussian_blur(
    matrix: &DMatrix<u8>,
    kernel: &DMatrix<f32>,
) -> DMatrix<u8> {
    apply_convolution(matrix, kernel.as_view())
}

pub fn sobel_blur(matrix: &DMatrix<u8>) -> DMatrix<u8> {
    let sobel_x_kernel =
        Matrix3::new(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
    let sobel_y_kernel =
        Matrix3::new(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);

    let sobel_x = apply_convolution(matrix, sobel_x_kernel.as_view());
    let sobel_y = apply_convolution(matrix, sobel_y_kernel.as_view());

    // Комбинируем результаты оператора Собеля
    let mut sobel_combined = DMatrix::zeros(matrix.nrows(), matrix.ncols());
    for y in 0..sobel_combined.nrows() {
        for x in 0..sobel_combined.ncols() {
            let value = ((sobel_x[(y, x)] as f32).powi(2)
                + (sobel_y[(y, x)] as f32).powi(2))
            .sqrt();
            sobel_combined[(y, x)] = value.clamp(0.0, 255.0) as u8;
        }
    }

    sobel_combined
}

fn apply_convolution(
    matrix: &DMatrix<u8>,
    kernel: DMatrixView<f32>,
) -> DMatrix<u8> {
    assert!(kernel.is_square());

    let (height, width) = (matrix.nrows(), matrix.ncols());

    let kernel_size = kernel.nrows();
    let kernel_radius = kernel_size / 2;
    let kernel_sum = kernel.sum();

    let mut result = DMatrix::zeros(height, width);

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
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

            result[(y, x)] = (sum / kernel_sum).clamp(0.0, 255.0) as u8;
            // result[(y, x)] = sum.clamp(0.0, 255.0) as u8;
        }
    }
    result
}
