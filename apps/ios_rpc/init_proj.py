import argparse
import re

default_team_id = "3FR42MXLK9"
default_bundle_identifier = 'ml.dmlc.tvmrpc'

parser = argparse.ArgumentParser(description='Update tvmrpc.xcodeproj\
 developer information')
parser.add_argument('--team_id', type=str, required=True,
                    help='Apple Developer Team ID.\n\
                    Can be found here:\n\
                    \n\
                    https://developer.apple.com/account/#/membership\n\
                    (example: {})'.format(default_team_id))

parser.add_argument('--bundle_identifier', type=str, required=False,
                    default=default_bundle_identifier,
                    help='The new bundle identifier\n\
                    (example: {})'.format(default_bundle_identifier))

args = parser.parse_args()
team_id = args.team_id
bundle_identifier = args.bundle_identifier

fi = open("tvmrpc.xcodeproj/project.pbxproj")
proj_config = fi.read()
fi.close()

proj_config = proj_config.replace(default_team_id, team_id)
proj_config = proj_config.replace(default_bundle_identifier, bundle_identifier)
fo = open("tvmrpc.xcodeproj/project.pbxproj", "w")
fo.write(proj_config)
fo.close()
