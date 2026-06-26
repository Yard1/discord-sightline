#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use super::{BatchHashError, collect_image_paths, sanitize_file_stem};
use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use std::{fs, path::Path};

#[derive(Debug, Serialize)]
pub(super) struct ImageAugmentSummary {
    input_path: String,
    output_dir: String,
    profile: &'static str,
    jpeg_quality: u8,
    source_count: usize,
    variants_written: usize,
    failed: usize,
    files: Vec<ImageAugmentFile>,
    errors: Vec<BatchHashError>,
}

#[derive(Debug, Serialize)]
struct ImageAugmentFile {
    source_path: String,
    output_path: String,
    variant: &'static str,
    width: u32,
    height: u32,
    bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ImageAugmentOptions {
    profile: ImageAugmentProfile,
    jpeg_quality: u8,
    include_original: bool,
}

#[derive(Debug, Clone, Copy)]
enum ImageAugmentProfile {
    Mild,
    Geometry,
    Full,
}

impl ImageAugmentProfile {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Mild => "mild",
            Self::Geometry => "geometry",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ImageAugmentTransform {
    Original,
    Rotate {
        degrees: f32,
    },
    CenterZoom {
        scale: f32,
    },
    ShearX {
        factor: f32,
    },
    ShearY {
        factor: f32,
    },
    PerspectiveX {
        top_scale: f32,
        bottom_scale: f32,
        shift: f32,
    },
    PerspectiveY {
        left_scale: f32,
        right_scale: f32,
        shift: f32,
    },
}

impl ImageAugmentTransform {
    fn label(self) -> &'static str {
        match self {
            Self::Original => "original_reencoded",
            Self::Rotate { degrees } if degrees < 0.0 && degrees.abs() <= 7.5 => "rotate_neg7",
            Self::Rotate { degrees } if degrees < 0.0 => "rotate_neg11",
            Self::Rotate { degrees } if degrees <= 7.5 => "rotate_pos7",
            Self::Rotate { .. } => "rotate_pos11",
            Self::CenterZoom { .. } => "center_crop_104",
            Self::ShearX { factor } if factor < 0.0 && factor.abs() <= 0.10 => "affine_shear_left",
            Self::ShearX { factor } if factor < 0.0 => "affine_shear_left_12",
            Self::ShearX { factor } if factor <= 0.10 => "affine_shear_right",
            Self::ShearX { .. } => "affine_shear_right_12",
            Self::ShearY { factor } if factor < 0.0 && factor.abs() <= 0.06 => "affine_shear_up",
            Self::ShearY { factor } if factor < 0.0 => "affine_shear_up_08",
            Self::ShearY { factor } if factor <= 0.06 => "affine_shear_down",
            Self::ShearY { .. } => "affine_shear_down_08",
            Self::PerspectiveX {
                top_scale,
                bottom_scale,
                ..
            } if top_scale < bottom_scale && top_scale >= 0.88 => "perspective_top_narrow",
            Self::PerspectiveX {
                top_scale,
                bottom_scale,
                ..
            } if top_scale < bottom_scale => "perspective_top_narrow_strong",
            Self::PerspectiveX { top_scale, .. } if top_scale <= 1.05 => {
                "perspective_bottom_narrow"
            }
            Self::PerspectiveX { .. } => "perspective_bottom_narrow_strong",
            Self::PerspectiveY {
                left_scale,
                right_scale,
                ..
            } if left_scale < right_scale && left_scale >= 0.88 => "perspective_left_narrow",
            Self::PerspectiveY {
                left_scale,
                right_scale,
                ..
            } if left_scale < right_scale => "perspective_left_narrow_strong",
            Self::PerspectiveY { left_scale, .. } if left_scale <= 1.05 => {
                "perspective_right_narrow"
            }
            Self::PerspectiveY { .. } => "perspective_right_narrow_strong",
        }
    }
}

pub(super) fn parse_augment_options(args: &[String]) -> Result<ImageAugmentOptions> {
    let mut options = ImageAugmentOptions {
        profile: ImageAugmentProfile::Geometry,
        jpeg_quality: 86,
        include_original: false,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--profile" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| anyhow!("--profile requires one of: mild, geometry, full"))?;
                options.profile = match value.as_str() {
                    "mild" => ImageAugmentProfile::Mild,
                    "geometry" => ImageAugmentProfile::Geometry,
                    "full" => ImageAugmentProfile::Full,
                    _ => return Err(anyhow!("--profile requires one of: mild, geometry, full")),
                };
                index += 2;
            }
            "--jpeg-quality" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| anyhow!("--jpeg-quality requires a value from 1 to 100"))?;
                options.jpeg_quality = value
                    .parse::<u8>()
                    .with_context(|| format!("parsing --jpeg-quality {value}"))?;
                anyhow::ensure!(
                    (1..=100).contains(&options.jpeg_quality),
                    "--jpeg-quality must be between 1 and 100"
                );
                index += 2;
            }
            "--include-original" => {
                options.include_original = true;
                index += 1;
            }
            other => return Err(anyhow!("unknown augment-images option {other}")),
        }
    }
    Ok(options)
}

pub(super) fn augment_image_batch(
    input_path: &Path,
    output_dir: &Path,
    options: ImageAugmentOptions,
) -> Result<ImageAugmentSummary> {
    fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "creating augmented image output directory {}",
            output_dir.display()
        )
    })?;

    let image_paths = collect_image_paths(input_path)?;
    let mut files = Vec::new();
    let mut errors = Vec::new();

    for (source_index, path) in image_paths.iter().enumerate() {
        match augment_image_file(path, source_index, output_dir, options) {
            Ok(mut written) => files.append(&mut written),
            Err(source) => errors.push(BatchHashError {
                source_path: path.display().to_string(),
                error: source.to_string(),
            }),
        }
    }

    Ok(ImageAugmentSummary {
        input_path: input_path.display().to_string(),
        output_dir: output_dir.display().to_string(),
        profile: options.profile.as_str(),
        jpeg_quality: options.jpeg_quality,
        source_count: image_paths.len(),
        variants_written: files.len(),
        failed: errors.len(),
        files,
        errors,
    })
}

fn augment_image_file(
    path: &Path,
    source_index: usize,
    output_dir: &Path,
    options: ImageAugmentOptions,
) -> Result<Vec<ImageAugmentFile>> {
    let source = ::image::open(path)
        .with_context(|| format!("decoding {}", path.display()))?
        .to_rgb8();
    let transforms = augment_transforms(options.profile, options.include_original);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_file_stem)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "image".to_owned());
    let mut files = Vec::with_capacity(transforms.len());

    for transform in transforms {
        let variant = apply_augment_transform(&source, transform);
        let output_path = output_dir.join(format!(
            "{:04}_{stem}__{}.jpg",
            source_index + 1,
            transform.label()
        ));
        write_jpeg(&variant, &output_path, options.jpeg_quality)
            .with_context(|| format!("writing {}", output_path.display()))?;
        let metadata = fs::metadata(&output_path)
            .with_context(|| format!("reading {}", output_path.display()))?;
        files.push(ImageAugmentFile {
            source_path: path.display().to_string(),
            output_path: output_path.display().to_string(),
            variant: transform.label(),
            width: variant.width(),
            height: variant.height(),
            bytes: metadata.len(),
        });
    }

    Ok(files)
}

fn augment_transforms(
    profile: ImageAugmentProfile,
    include_original: bool,
) -> Vec<ImageAugmentTransform> {
    let mut transforms = Vec::new();
    if include_original {
        transforms.push(ImageAugmentTransform::Original);
    }
    match profile {
        ImageAugmentProfile::Mild => {
            transforms.push(ImageAugmentTransform::Rotate { degrees: -5.0 });
            transforms.push(ImageAugmentTransform::Rotate { degrees: 5.0 });
            transforms.push(ImageAugmentTransform::CenterZoom { scale: 1.035 });
        }
        ImageAugmentProfile::Geometry => {
            transforms.push(ImageAugmentTransform::Rotate { degrees: -7.0 });
            transforms.push(ImageAugmentTransform::Rotate { degrees: 7.0 });
            transforms.push(ImageAugmentTransform::CenterZoom { scale: 1.04 });
            transforms.push(ImageAugmentTransform::ShearX { factor: -0.08 });
            transforms.push(ImageAugmentTransform::ShearX { factor: 0.08 });
            transforms.push(ImageAugmentTransform::ShearY { factor: -0.05 });
            transforms.push(ImageAugmentTransform::ShearY { factor: 0.05 });
            transforms.push(ImageAugmentTransform::PerspectiveX {
                top_scale: 0.90,
                bottom_scale: 1.04,
                shift: 0.035,
            });
            transforms.push(ImageAugmentTransform::PerspectiveX {
                top_scale: 1.04,
                bottom_scale: 0.90,
                shift: -0.035,
            });
            transforms.push(ImageAugmentTransform::PerspectiveY {
                left_scale: 0.90,
                right_scale: 1.04,
                shift: 0.030,
            });
            transforms.push(ImageAugmentTransform::PerspectiveY {
                left_scale: 1.04,
                right_scale: 0.90,
                shift: -0.030,
            });
        }
        ImageAugmentProfile::Full => {
            transforms.extend(augment_transforms(ImageAugmentProfile::Geometry, false));
            transforms.push(ImageAugmentTransform::Rotate { degrees: -11.0 });
            transforms.push(ImageAugmentTransform::Rotate { degrees: 11.0 });
            transforms.push(ImageAugmentTransform::ShearX { factor: -0.12 });
            transforms.push(ImageAugmentTransform::ShearX { factor: 0.12 });
            transforms.push(ImageAugmentTransform::ShearY { factor: -0.08 });
            transforms.push(ImageAugmentTransform::ShearY { factor: 0.08 });
            transforms.push(ImageAugmentTransform::PerspectiveX {
                top_scale: 0.82,
                bottom_scale: 1.08,
                shift: 0.055,
            });
            transforms.push(ImageAugmentTransform::PerspectiveX {
                top_scale: 1.08,
                bottom_scale: 0.82,
                shift: -0.055,
            });
            transforms.push(ImageAugmentTransform::PerspectiveY {
                left_scale: 0.82,
                right_scale: 1.08,
                shift: 0.050,
            });
            transforms.push(ImageAugmentTransform::PerspectiveY {
                left_scale: 1.08,
                right_scale: 0.82,
                shift: -0.050,
            });
        }
    }
    transforms
}

fn apply_augment_transform(
    source: &::image::RgbImage,
    transform: ImageAugmentTransform,
) -> ::image::RgbImage {
    let width = source.width();
    let height = source.height();
    let fill = estimate_canvas_fill(source);
    let mut output = ::image::RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let (source_x, source_y) =
                inverse_augmented_coordinate(x as f32, y as f32, width, height, transform);
            let pixel = sample_rgb_bilinear(source, source_x, source_y, fill);
            output.put_pixel(x, y, pixel);
        }
    }
    output
}

fn estimate_canvas_fill(source: &::image::RgbImage) -> ::image::Rgb<u8> {
    let width = source.width();
    let height = source.height();
    if width == 0 || height == 0 {
        return ::image::Rgb([0, 0, 0]);
    }

    let samples = 16_u32.min(width.max(height));
    let mut sums = [0_u64; 3];
    let mut count = 0_u64;
    for index in 0..samples {
        let x = scaled_edge_index(index, samples, width);
        let y = scaled_edge_index(index, samples, height);
        accumulate_rgb(*source.get_pixel(x, 0), &mut sums, &mut count);
        accumulate_rgb(*source.get_pixel(x, height - 1), &mut sums, &mut count);
        accumulate_rgb(*source.get_pixel(0, y), &mut sums, &mut count);
        accumulate_rgb(*source.get_pixel(width - 1, y), &mut sums, &mut count);
    }

    ::image::Rgb([
        (sums[0] / count.max(1)) as u8,
        (sums[1] / count.max(1)) as u8,
        (sums[2] / count.max(1)) as u8,
    ])
}

fn scaled_edge_index(index: u32, samples: u32, size: u32) -> u32 {
    if samples <= 1 || size <= 1 {
        return 0;
    }
    (index * (size - 1)) / (samples - 1)
}

fn accumulate_rgb(pixel: ::image::Rgb<u8>, sums: &mut [u64; 3], count: &mut u64) {
    for (sum, value) in sums.iter_mut().zip(pixel.0) {
        *sum += u64::from(value);
    }
    *count += 1;
}

fn inverse_augmented_coordinate(
    x: f32,
    y: f32,
    width: u32,
    height: u32,
    transform: ImageAugmentTransform,
) -> (f32, f32) {
    let center_x = (width.saturating_sub(1)) as f32 * 0.5;
    let center_y = (height.saturating_sub(1)) as f32 * 0.5;
    let dx = x - center_x;
    let dy = y - center_y;
    match transform {
        ImageAugmentTransform::Original => (x, y),
        ImageAugmentTransform::Rotate { degrees } => {
            let radians = degrees.to_radians();
            let cos = radians.cos();
            let sin = radians.sin();
            (
                center_x + cos.mul_add(dx, sin * dy),
                center_y + (-sin).mul_add(dx, cos * dy),
            )
        }
        ImageAugmentTransform::CenterZoom { scale } => {
            (center_x + dx / scale, center_y + dy / scale)
        }
        ImageAugmentTransform::ShearX { factor } => (x - factor * dy, y),
        ImageAugmentTransform::ShearY { factor } => (x, y - factor * dx),
        ImageAugmentTransform::PerspectiveX {
            top_scale,
            bottom_scale,
            shift,
        } => {
            let denom = height.saturating_sub(1).max(1) as f32;
            let y01 = y / denom;
            let scale = top_scale + (bottom_scale - top_scale) * y01;
            let centered_y = y01 - 0.5;
            let shifted_x = x - shift * centered_y * width as f32;
            (center_x + (shifted_x - center_x) / scale, y)
        }
        ImageAugmentTransform::PerspectiveY {
            left_scale,
            right_scale,
            shift,
        } => {
            let denom = width.saturating_sub(1).max(1) as f32;
            let x01 = x / denom;
            let scale = left_scale + (right_scale - left_scale) * x01;
            let centered_x = x01 - 0.5;
            let shifted_y = y - shift * centered_x * height as f32;
            (x, center_y + (shifted_y - center_y) / scale)
        }
    }
}

fn sample_rgb_bilinear(
    source: &::image::RgbImage,
    x: f32,
    y: f32,
    fill: ::image::Rgb<u8>,
) -> ::image::Rgb<u8> {
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 {
        return fill;
    }
    let max_x = source.width().saturating_sub(1) as f32;
    let max_y = source.height().saturating_sub(1) as f32;
    if x > max_x || y > max_y {
        return fill;
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(source.width().saturating_sub(1));
    let y1 = (y0 + 1).min(source.height().saturating_sub(1));
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let p00 = source.get_pixel(x0, y0).0;
    let p10 = source.get_pixel(x1, y0).0;
    let p01 = source.get_pixel(x0, y1).0;
    let p11 = source.get_pixel(x1, y1).0;
    let mut out = [0_u8; 3];
    for channel in 0..3 {
        let top = f32::from(p00[channel]).mul_add(1.0 - tx, f32::from(p10[channel]) * tx);
        let bottom = f32::from(p01[channel]).mul_add(1.0 - tx, f32::from(p11[channel]) * tx);
        let value = top.mul_add(1.0 - ty, bottom * ty).round().clamp(0.0, 255.0);
        out[channel] = value as u8;
    }
    ::image::Rgb(out)
}

fn write_jpeg(image: &::image::RgbImage, output_path: &Path, quality: u8) -> Result<()> {
    let mut bytes = Vec::new();
    let dynamic = ::image::DynamicImage::ImageRgb8(image.clone());
    let mut encoder = ::image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, quality);
    encoder
        .encode_image(&dynamic)
        .context("encoding JPEG variant")?;
    fs::write(output_path, bytes).with_context(|| format!("writing {}", output_path.display()))
}
