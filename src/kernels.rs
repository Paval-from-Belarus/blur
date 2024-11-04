use nalgebra::DMatrix;

pub fn gaussian(radius: usize) -> DMatrix<f32> {
    let size = radius * 2 + 1;
    let sigma = radius as f32 / 3.0;
    let mean = radius as f32;
    let mut kernel = DMatrix::zeros(size, size);
    let mut sum = 0.0;

    // Fill in the kernel with Gaussian values
    for y in 0..size {
        for x in 0..size {
            let exponent = -(((x as f32 - mean).powi(2)
                + (y as f32 - mean).powi(2))
                / (2.0 * sigma.powi(2)));
            kernel[(y, x)] = (1.0
                / (2.0 * std::f32::consts::PI * sigma.powi(2)))
                * exponent.exp();
            sum += kernel[(y, x)];
        }
    }

    // Normalize the kernel
    for y in 0..size {
        for x in 0..size {
            kernel[(y, x)] /= sum;
        }
    }

    kernel
}

pub fn box_kernel(radius: usize) -> DMatrix<f32> {
    let size = 2 * radius;
    DMatrix::from_element(size, size, 1.0 / (size * size) as f32)
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbossKind {
    Left,
    Right,
    Edge,
    Application,
}

#[rustfmt::skip]
pub fn emboss(kind: EmbossKind) -> DMatrix<f32> {
    match kind {
        EmbossKind::Left => DMatrix::from_row_slice(3, 3, &[
            0.0, 0.0, -1.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
        ]),
        EmbossKind::Right => DMatrix::from_row_slice(3, 3, &[
            -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        ]),
        EmbossKind::Edge => DMatrix::from_row_slice(5, 5, &[
            -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0,
        ]),
        EmbossKind::Application => DMatrix::from_row_slice(3, 3, &[
            -2.0, -1.0, 0.0,
            -1.0, 1.0, 1.0,
            0.0, 1.0, 2.0

        ]),
    }
}
