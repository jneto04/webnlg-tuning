import glob
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd

def _buid_csv_dataset(split):
  files = glob.glob("/home/joaquimneto04/data/release_v3.0/en/"+split+"/**/*.xml", recursive=True)
  triple_re=re.compile('(\d)triples')
  data_dct={}

  for f in files:
      tree = ET.parse(f)
      root = tree.getroot()
      triples_num=int(triple_re.findall(f)[0])
      for sub_root in root:
          for ss_root in sub_root:
              strutured_master=[]
              unstructured=[]
              for entry in ss_root:
                  unstructured.append(entry.text)
                  strutured=[triple.text for triple in entry]
                  strutured_master.extend(strutured)
              unstructured=[i for i in unstructured if i.replace('\n','').strip()!='' ]
              strutured_master=strutured_master[-triples_num:]
              strutured_master_str=(' && ').join(strutured_master)
              data_dct[strutured_master_str]=unstructured
  mdata_dct={"prefix":[], "input_text":[], "target_text":[]}

  for st,unst in data_dct.items():
      for i in unst:
          mdata_dct['prefix'].append('webNLG')
          mdata_dct['input_text'].append(st+' </s>')
          mdata_dct['target_text'].append(i+' </s>')


  df=pd.DataFrame(mdata_dct)
  df.to_csv('/home/joaquimneto04/WebNLG/webNLG2020_'+split+'.csv')

_buid_csv_dataset('train')
_buid_csv_dataset('test')
_buid_csv_dataset('dev')
