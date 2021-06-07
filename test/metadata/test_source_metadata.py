import unittest
from jsonschema import ValidationError
from mobie.validation.utils import validate_with_schema


class TestSourceMetadata(unittest.TestCase):
    def test_image_source(self):
        from mobie.metadata import get_image_metadata

        name = 'my-image'
        source = get_image_metadata(name, "/path/to/bdv.xml")
        validate_with_schema(source, 'source')

        source = get_image_metadata(name, "/path/to/bdv.xml")
        source["image"]["imageData"] = {"format": "bdv.n5",
                                        "relativePath": "path/to/bdv.xml",
                                        "s3Store": "https://s3.com/bdv.xml"}
        validate_with_schema(source, 'source')

        # check missing fields
        source = get_image_metadata(name, "/path/to/bdv.xml")
        source["image"].pop("imageData")
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        source = get_image_metadata(name, "/path/to/bdv.xml")
        source["image"]["imageData"].pop("format")
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        source = get_image_metadata(name, "/path/to/bdv.xml")
        source["image"]["imageData"].pop("relativePath")
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        # check invalid fields
        source = get_image_metadata(name, "/path/to/bdv.xml")
        source["foo"] = "bar"
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        source = get_image_metadata(name, "/path/to/bdv.xml")
        source["image"]["foo"] = "bar"
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        source = get_image_metadata(name, "/path/to/bdv.xml")
        source["image"]["imageData"]["foo"] = "bar"
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        source = get_image_metadata(name, "/path/to/bdv.xml")
        source["image"]["imageData"]["format"] = "tif"
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

    def test_segmentation_source(self):
        from mobie.metadata import get_segmentation_metadata

        name = 'my-segmentation'
        source = get_segmentation_metadata(name, "/path/to/bdv.xml")
        validate_with_schema(source, 'source')

        source = get_segmentation_metadata(name, "/path/to/bdv.xml",
                                           table_location="/path/to/tables")
        validate_with_schema(source, 'source')

        # check missing fields
        source = get_segmentation_metadata(name, "/path/to/bdv.xml")
        source["segmentation"].pop("imageData")
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        # check invalid fields
        source = get_segmentation_metadata(name, "/path/to/bdv.xml")
        source["foo"] = "bar"
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        source = get_segmentation_metadata(name, "/path/to/bdv.xml")
        source["segmentation"]["foo"] = "bar"
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        source = get_segmentation_metadata(name, "/path/to/bdv.xml")
        source["segmentation"]["imageData"]["foo"] = "bar"
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        source = get_segmentation_metadata(name, "/path/to/bdv.xml",
                                           table_location="/path/to/tables")
        source["segmentation"]["tableData"]["format"] = "excel"
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')

        source = get_segmentation_metadata(name, "/path/to/bdv.xml",
                                           table_location="/path/to/tables")
        source["segmentation"]["tableData"]["foo"] = "bar"
        with self.assertRaises(ValidationError):
            validate_with_schema(source, 'source')


if __name__ == '__main__':
    unittest.main()
