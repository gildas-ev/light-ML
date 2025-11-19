import uuid
from xml.etree import ElementTree as ET

class Roc:
    _CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"
    _RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    _XML_ATTRIBUTE_MAP = {
        "Exposure2012": "Exposure2012",
        "Contrast2012": "Contrast2012",
        "Highlights2012": "Highlights2012",
        "Shadows2012": "Shadows2012",
        "Whites2012": "Whites2012",
        "Blacks2012": "Blacks2012",
        "Vibrance": "Vibrance",
        "Dehaze": "Dehaze",
    }

    def __init__(
        self,
        Exposure2012=0,
        Contrast2012=0,
        Highlights2012=0,
        Shadows2012=0,
        Whites2012=0,
        Blacks2012=0,
        Vibrance=0,
        Dehaze=0,
    ):
        self.exposure2012 = Exposure2012
        self.contrast2012 = Contrast2012
        self.highlights2012 = Highlights2012
        self.shadows2012 = Shadows2012
        self.whites2012 = Whites2012
        self.blacks2012 = Blacks2012
        self.vibrance = Vibrance
        self.dehaze = Dehaze
    
    def to_dict(self) -> dict[str, int | float]:
        return {
            "Exposure2012": self.exposure2012,
            "Contrast2012": self.contrast2012,
            "Highlights2012": self.highlights2012,
            "Shadows2012": self.shadows2012,
            "Whites2012": self.whites2012,
            "Blacks2012": self.blacks2012,
            "Vibrance": self.vibrance,
            "Dehaze": self.dehaze,
        }

    def to_xmp(
        self,
        name: str | None = None,
        preset_uuid: str | None = None,
    ):
        uid = (preset_uuid or uuid.uuid4().hex).upper()
        preset_name = name or f"predicted_preset_{uid}"
        xmp = f'''<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        ">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   crs:PresetType="Normal"
   crs:Cluster=""
   crs:UUID="{uid}"
   crs:SupportsAmount2="False"
   crs:SupportsAmount="False"
   crs:SupportsColor="True"
   crs:SupportsMonochrome="True"
   crs:SupportsHighDynamicRange="True"
   crs:SupportsNormalDynamicRange="True"
   crs:SupportsSceneReferred="True"
   crs:SupportsOutputReferred="True"
   crs:RequiresRGBTables="False"
   crs:CameraModelRestriction=""
   crs:Copyright=""
   crs:ContactInfo=""
   crs:Version="17.0"
   crs:ProcessVersion="15.4"
    crs:WhiteBalance="As Shot"
   crs:Exposure2012="{self.exposure2012}"
   crs:Contrast2012="{self.contrast2012}"
   crs:Highlights2012="{self.highlights2012}"
   crs:Shadows2012="{self.shadows2012}"
   crs:Whites2012="{self.whites2012}"
   crs:Blacks2012="{self.blacks2012}"
   crs:Texture="0"
   crs:Clarity2012="0"
   crs:Dehaze="{self.dehaze}"
   crs:Vibrance="{self.vibrance}"
   crs:Saturation="0"
   crs:ConvertToGrayscale="False"
   crs:CameraProfile="Default Color"
   crs:HasSettings="True">
   <crs:Name>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{preset_name}</rdf:li>
    </rdf:Alt>
   </crs:Name>
   <crs:ShortName>
    <rdf:Alt>
     <rdf:li xml:lang="x-default"/>
    </rdf:Alt>
   </crs:ShortName>
   <crs:SortName>
    <rdf:Alt>
     <rdf:li xml:lang="x-default"/>
    </rdf:Alt>
   </crs:SortName>
   <crs:Group>
    <rdf:Alt>
     <rdf:li xml:lang="x-default"/>
    </rdf:Alt>
   </crs:Group>
   <crs:Description>
    <rdf:Alt>
     <rdf:li xml:lang="x-default"/>
    </rdf:Alt>
   </crs:Description>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
'''
        return xmp

    @classmethod
    def from_xmp(cls, xmp_str: str) -> "Roc":
        try:
            root = ET.fromstring(xmp_str)
        except ET.ParseError as exc:
            raise ValueError("Invalid XMP content") from exc

        description = root.find(f".//{{{cls._RDF_NS}}}Description")
        if description is None:
            raise ValueError("Could not find rdf:Description in XMP data")

        kwargs = {}
        for init_name, xml_name in cls._XML_ATTRIBUTE_MAP.items():
            attr_key = f"{{{cls._CRS_NS}}}{xml_name}"
            raw_value = description.attrib.get(attr_key)
            if raw_value is None:
                continue
            kwargs[init_name] = cls._convert_value(raw_value)

        return cls(**kwargs)

    @staticmethod
    def _convert_value(value : str) -> int | float:
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported numeric value: {value}") from exc
        if number.is_integer():
            return int(number)
        return number

    def __str__(self):
        output = ""
        attributes = [
            ("Exposure2012", self.exposure2012),
            ("Contrast2012", self.contrast2012),
            ("Highlights2012", self.highlights2012),
            ("Shadows2012", self.shadows2012),
            ("Whites2012", self.whites2012),
            ("Blacks2012", self.blacks2012),
            ("Vibrance", self.vibrance),
            ("Dehaze", self.dehaze),
        ]
        for label, value in attributes:
            output += f"{label}: {value}\n"
        return output

    def __add__(self, roc2):
        return Roc(
            Exposure2012=self.exposure2012 + roc2.exposure2012,
            Contrast2012=self.contrast2012 + roc2.contrast2012,
            Highlights2012=self.highlights2012 + roc2.highlights2012,
            Shadows2012=self.shadows2012 + roc2.shadows2012,
            Whites2012=self.whites2012 + roc2.whites2012,
            Blacks2012=self.blacks2012 + roc2.blacks2012,
            Vibrance=self.vibrance + roc2.vibrance,
            Dehaze=self.dehaze + roc2.dehaze,
        )
    
    def __sub__(self, roc2):
        return Roc(
            Exposure2012=self.exposure2012 - roc2.exposure2012,
            Contrast2012=self.contrast2012 - roc2.contrast2012,
            Highlights2012=self.highlights2012 - roc2.highlights2012,
            Shadows2012=self.shadows2012 - roc2.shadows2012,
            Whites2012=self.whites2012 - roc2.whites2012,
            Blacks2012=self.blacks2012 - roc2.blacks2012,
            Vibrance=self.vibrance - roc2.vibrance,
            Dehaze=self.dehaze - roc2.dehaze,
        )
