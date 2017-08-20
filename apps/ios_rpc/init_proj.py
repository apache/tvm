import argparse
import re

parser = argparse.ArgumentParser(description='Update tvmrpc.xcodeproj\
 developer information')
parser.add_argument('--org_unit', type=str, required=True,
                    help='Your own Organization Unit.\n\
                    The Organization Unit can be found by following:\n\
                    1. Open Keychain Access.\n\
                    2. Find out your own iPhone Developer certificate.\n\
                    3. Right click certificate, choose ```Get Info```.\n\
                    4. Read & copy your Organization Unit.')

parser.add_argument('--bundle_identifier', type=str, required=False,
                    default="tvmrpc",
                    help='The new bundle identifier')

args = parser.parse_args()
org_unit = args.org_unit
bundle_identifier = args.bundle_identifier

fi = open("tvmrpc.xcodeproj/project.pbxproj")
proj_config = fi.read()
fi.close()

proj_config = proj_config.replace("3FR42MXLK9", org_unit)
proj_config = proj_config.replace("ml.dmlc.tvmrpc", bundle_identifier)
fo = open("tvmrpc.xcodeproj/project.pbxproj", "w")
fo.write(proj_config)
fo.close()