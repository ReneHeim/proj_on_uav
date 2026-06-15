from src.pipeline_extract_data import remove_images_already_processed


def test_remove_images_supports_exact_input_path(tmp_path):
    image = tmp_path / "IMG_0004_6.tif"
    image.touch()
    output = tmp_path / "output"
    output.mkdir()

    remaining = remove_images_already_processed(str(image), output)

    assert remaining == [image]


def test_remove_images_skips_processed_image(tmp_path):
    image = tmp_path / "IMG_0004_6.tif"
    image.touch()
    output = tmp_path / "output"
    output.mkdir()
    (output / "week_0_IMG_0004_6.tif.parquet").touch()

    remaining = remove_images_already_processed(str(tmp_path / "*.tif"), output)

    assert remaining == []
