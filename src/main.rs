pub mod kernels;
mod operator;

use std::time::Instant;

use image::open;
use kernels::EmbossKind;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub box_radius: usize,
    pub gaussian_radius: usize,
    pub median_radius: usize,
    pub emboss_kind: EmbossKind,
    pub image: String,
}

fn main() {
    let raw_config =
        std::fs::read_to_string("config.toml").expect("Config is not found");

    let config =
        toml::from_str::<Config>(&raw_config).expect("Failed to parse config");

    let img = open(&config.image)
        .expect("Failed to open image")
        .into_rgba8();

    let operator = operator::Operator::from_rgba(&img);

    operator
        .box_blur(config.box_radius)
        .to_image()
        .save("box_blur.png")
        .unwrap();

    operator
        .gaussian_blur(config.gaussian_radius)
        .to_image()
        .save("gaussian_blur.png")
        .unwrap();

    operator
        .sobel_blur()
        .to_image()
        .save("sobel_blur.png")
        .unwrap();

    let time = Instant::now();

    operator
        .median(config.median_radius)
        .to_image()
        .save("median_blur.png")
        .unwrap();

    println!("Elapsed: {}", time.elapsed().as_millis());

    operator
        .emboss(config.emboss_kind)
        .to_image()
        .save("emboss.png")
        .unwrap();

    println!("Images are saved");
}
