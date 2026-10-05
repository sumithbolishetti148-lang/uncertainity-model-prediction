import pytest
from io import BytesIO
from PIL import Image

from frontend.uploads import validate_ecg_preview, validate_image_preview


def test_upload_content_validation():
    signal_csv = (",".join(str(i) for i in range(16)) + "\n" +
                  ",".join(str(i + 1) for i in range(16))).encode()
    assert "numeric" in validate_ecg_preview("trace.csv", signal_csv)
    with pytest.raises(ValueError, match="at least 16"):
        validate_ecg_preview("short.csv", b"1,2\n3,4\n")
    with pytest.raises(ValueError, match="no numeric"):
        validate_ecg_preview("bad.csv", b"abc")
    with pytest.raises(ValueError, match="Unsupported"):
        validate_ecg_preview("bad.exe", b"anything")
    with pytest.raises(ValueError, match="10 MB"):
        validate_ecg_preview("large.csv", b"0" * (10 * 1024 * 1024 + 1))
    image = BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    assert "valid" in validate_image_preview("image.png", image.getvalue())
    with pytest.raises(ValueError, match="corrupted"):
        validate_image_preview("bad.png", b"not an image")
