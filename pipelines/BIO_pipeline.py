# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import yaml
import os
import xml.etree.ElementTree as ET
from modules.Training_utilities import XML_BIO_converter
from datetime import datetime
import pprint

""" Define XML parsing method """
class XML_BIO_LKW_converter(XML_BIO_converter):
  def _parse_XML(self, xml_filename:str):
    """
    This method inputs a xml_filename with dir
    outputs a text content. 
    If has_tag == True, outputs text content + list of tags 4-tuple 
    (tag_id, tag_name, start, end)

    Parameters
    ----------
    xml_filename : str
      xml filename with dir.
    """
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    text = root.find('TEXT').text
    
    tags = []
    for tag in root.find('TAGS'):
      if tag.tag == 'HIGHLIGHT':
        continue
      tag_id = tag.attrib['id']
      tag_name = tag.tag
      start_pos, end_pos = tag.attrib['spans'].split('~')
      if tag_name in ['CONSULT_DATETIME', 'LKW']:
        att_type = tag.attrib['Type'] 
      elif tag_name in ['tPA', 'TRANSPORTATION']:
        att_type = tag.attrib['Modality'] 
        
      tags.append((tag_id, f"{tag_name}_{att_type}", int(start_pos), int(end_pos)))
    
    return text, tags
  
""" Pipeline """
def main():
  print('BIO pipeline starts:')
  print(datetime.now())
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))
    
  print('Config loaded:')
  pprint.pprint(config)
  print(datetime.now())
  converter = XML_BIO_LKW_converter(xml_dir=config['XML_dir'],
                                BIO_dir=config['BIO_dir'],
                                mode=config['BIO_mode'])
  
  converter.pop_BIO()

if __name__ == '__main__':
  main()