from pathlib import Path


def test_static_class_mapping_includes_bucket_and_golf_cart():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "automatic_annotation"
        / "src"
        / "gen_prompt_point.cpp"
    )
    source = source_path.read_text(encoding="utf-8")

    assert 'class_name_to_label_["Bucket"] = "Bucket";' in source
    assert 'class_name_to_label_["Golf_Cart"] = "Golf_Cart";' in source
    assert 'class_name_to_id_["Bucket"] = 5;' in source
    assert 'class_name_to_id_["Golf_Cart"] = 6;' in source
