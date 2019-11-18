import re
from urllib.parse import urlparse
from pathlib import Path
#
# d = Path("/some/path")
# print(d)
# print(d / "/repos".strip("/"))
# print(d / "lala")
# print((d / "lala").absolute())
#
# # print(urlparse("https://github.com/bioimage-io/pytorchbioimageio:configurations/readers/BroadNucleusData.reader.yaml:9fec4dd"))
# print(urlparse("torch.nn.Adam:lala.lulu"))
# # print(urlparse("file:./repos/configuration/models/UNet2dExample.model.yaml:lala"))
# # print(urlparse("file:c:/repos/configuration/models/UNet2dExample.model.yaml:lala"))
# # print(urlparse("./repos/configuration/models/UNet2dExample.model.yaml:lala"))
# # print(urlparse("c:/repos/configuration/models/UNet2dExample.model.yaml:lala"))
# print(urlparse("https://github.com/bioimage-io/pytorchbioimageio:configurations/readers/BroadNucleusData.reader.yaml:9fec4dd"))
# print(urlparse("github:bioimage-io/pytorchbioimageio:configurations/readers/BroadNucleusData.reader.yaml:9fec4dd"))
#
print(urlparse("https://github.com/bioimage-io/example-unet-configurations/blob/master/models/unet-2d-nuclei-broad/dummy_config_parser.py#L22"))
# # print(urlparse("http://dx.doi.org/10.1037/rmh0000008"))
from typing import Optional


for name in ["def class a:as"]:
    # print(name, bool(re.fullmatch("L[0-9]+", name)))
    match = re.match("((class)|(def)) (?P<obj_name>\D\S*):", name)
    print(name, bool(match))
    if match:
        print("\t", match.groups(), match.group("obj_name"))