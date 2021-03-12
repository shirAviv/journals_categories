import os
import pickle
from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch
from Bio import Entrez
import json
import pandas as pd
from pybliometrics.scopus import ScopusSearch



class Utils():
    def __init__(self):

        con_file = open("conf.json")
        config = json.load(con_file)
        con_file.close()
        self.path = config['path']

        ## Initialize client
        # self.client = ElsClient(config['apikey'])
        # self.client.inst_token = config['insttoken']

    def load_csv_data_to_df(self,name, delimiter=None):
        obj_path = os.path.join(self.path, name)
        papers = pd.read_csv(obj_path,keep_default_na=False, delimiter=delimiter)
        return papers

    def save_obj(self,obj, name):
        obj_path = os.path.join(self.path, name)
        # str=pickle.dumps(obj)
        with open(obj_path + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)

    def load_obj(self,name):
        obj_path = os.path.join(self.path, name)
        with open(obj_path + '.pkl', 'rb') as f:
            return pickle.load(f)

    def write_to_csv(self, df, name, index=False):
        file_path = os.path.join(self.path, name)
        df.to_csv(file_path, index=index)

    def read_from_csv(self,name):
        csv_path=os.path.join(self.path, name)
        with open(csv_path, encoding="utf8") as csv_file:
            contents = csv_file.read()
        return contents

    def get_data_from_doi(self, doi,title):
        id=None
        affil=None
        pub_name=None
        pub_type=None
        # try:
        try:
            doi_doc=ScopusSearch(doi, subscriber=False)
            if 'pubmed-id' in doi_doc._json[0].keys():
                id=doi_doc._json[0]["pubmed-id"]
            if 'affiliation' in doi_doc._json[0].keys():
                affil= doi_doc._json[0]['affiliation']
            pub_name = doi_doc._json[0]['prism:publicationName']
            pub_type = doi_doc._json[0]['subtypeDescription']
        except:
            print("failed with scopus")
        if id==None:
            doi_doc = FullDoc(doi=doi)
            if doi_doc.read(self.client):
                # print("doi_doc.title: ", doi_doc.title)
                doi_doc.write()
                pub_name=doi_doc.data['coredata']['prism:publicationName']
                if 'pubType' in doi_doc.data['coredata'].keys():
                    pub_type=str(doi_doc.data['coredata']['pubType']).strip()
            else:
                print("Read document failed. no id for doi {}. trying with title".format(doi))
                doi_doc=None
                 # return doi, affil
            id = None
            if doi_doc==None or (not 'pubmed-id' in doi_doc._data.keys()):
                print("trying with title")
                # try with title
                Entrez.email = 'shirAviv@protonmail.com'
                if doi_doc==None:
                    query = title
                else:
                    query = doi_doc.title
                handle = Entrez.esearch(db='pubmed',
                                        retmode='xml',
                                        term=query)
                results = Entrez.read(handle)
                if int(results['Count']) > 0:
                    id = results['IdList']
            else:
                id = doi_doc._data['pubmed-id']
        if id != None:
            return self.fetch_data_from_pubmed(id), affil, pub_name, pub_type

        else:
            print("no pubmed id found for doi {}".format(doi))
            return doi, affil, pub_name, pub_type
        # except:
        #     print("caught exception for doi {}".format(doi))
        #     return doi, affil, pub_name, pub_type

    def fetch_data_from_pubmed(self, id):
        Entrez.email = 'shirAviv@protonmail.com'
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=id)
        results = Entrez.read(handle)
        # print(results)
        return results
