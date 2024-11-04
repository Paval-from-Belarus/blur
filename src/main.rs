pub mod kernels;
mod operator;

use image::{open, RgbaImage};
use nalgebra::DMatrix;

fn image_to_matrix(img: &RgbaImage) -> DMatrix<u8> {
    let (width, height) = img.dimensions();
    let mut matrix = DMatrix::zeros(height as usize, width as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            matrix[(y as usize, x as usize)] = pixel[0]; // Используем только канал яркости
        }
    }

    matrix
}

fn matrix_to_image(matrix: &DMatrix<u8>) -> RgbaImage {
    let (height, width) = (matrix.nrows() as u32, matrix.ncols() as u32);
    let mut img = RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let value = matrix[(y as usize, x as usize)];
            img.put_pixel(x, y, image::Rgba([value, value, value, 255]));
        }
    }

    img
}

fn main() {
    let img = open("images/night_city.png")
        .expect("Failed to open image")
        .into_rgba8();

    let operator = operator::Operator::from_image(&img);

    operator
        .box_blur(3)
        .to_image()
        .save("box_blur.png")
        .unwrap();

    operator
        .gaussian_blur(5)
        .to_image()
        .save("gaussian_blur.png")
        .unwrap();

    operator
        .sobel_blur()
        .to_image()
        .save("sobel_blur.png")
        .unwrap();

    println!("Фильтры применены и изображения сохранены.");
}
