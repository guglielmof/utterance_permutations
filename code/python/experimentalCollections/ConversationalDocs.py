import lxml.etree as ET
from io import StringIO
#from xml.etree import XMLParser
#from bs4 import BeautifulSoup

#def importCAsTDocs():



#def getCAsTDocsByIDs():



def preprocessMSMARCO():
    with open('../../../EXPERIMENTAL_COLLECTIONS/CONVERSATIONAL/MSMARCO/collection.tsv.xml', 'r') as f:  # Reading file
        xml = f.read()

    xml = '<ROOT>' + xml + '</ROOT>'  # Let's add a root tag

    with open('../../../EXPERIMENTAL_COLLECTIONS/CONVERSATIONAL/MSMARCO/collection.tsv.xml', 'w') as f:  # Reading file
        f.write(xml)

def readMSMARCO():

    fullColl = {}
    path = '../../../EXPERIMENTAL_COLLECTIONS/CONVERSATIONAL/MSMARCO/collection.tsv.xml'
    # read the config file to memory

    with open(path, 'r') as fh:
        lines = fh.readlines()
        k = 0
        while k<len(lines):
            if "<DOCNO>" in lines[k]:
                dn = lines[k][7:-9]
                k = k+6
                bd = ""
                while "</BODY>" not in lines[k]:
                    bd += lines[k]
                    k+=1
                fullColl[dn] = bd
            k+=1

    return fullColl

def readCAR():

    fullColl = {}
    path = '../../../EXPERIMENTAL_COLLECTIONS/CONVERSATIONAL/CAR/dedup.articles-paragraphs.cbor.xml'
    # read the config file to memory

    with open(path, 'r') as fh:
        lines = fh.readlines()
        k = 0
        while k<len(lines):
            if "<DOCNO>" in lines[k]:
                dn = lines[k][7:-9]
                k = k+5
                bd = lines[k]
                fullColl[dn] = bd
            k+=1

    return fullColl

def iterate_xml(xmlfile):

    for event, element in ET.iterparse(xmlfile, recover=True, huge_tree=True, tag='DOC'):
        yield element


def readReducedCollection():
    reducedColl = {}
    with open("../../data/processed_collections/ds_reduced.tsv", "r") as F:
        for l in F.readlines():
            try:
                splitted = l.strip().split("\t")
                idx = splitted[0]
                txt = "\t".join(splitted[1:])
            except Exception as e:
                print(e)
                print(l)
                break
            reducedColl[idx] = txt
    return reducedColl