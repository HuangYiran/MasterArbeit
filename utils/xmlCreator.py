from collections import namedtuple
import xml.dom.minidom
import os

def createXmlForPrepro(dirs, header = './'):
    """
    input:
        dirs: list of string, contain the dirs that save the file to do the prepro
        header: the prefix of the dir.
    output:
        null
    condtions:
        1. there are 6 files in each dir: data_s1.en, data_s1.de, data_s2.en, data_s2.de, data_ref.en, data_ref.de
        2. dirname will be used as the itemname, so it is better if the diranem does not contain punctuations
    """
    # assertion, make sure that all the files exist
    for di in dirs:
        assert(os.path.exists(header+di+'/data_s1.en'))
        assert(os.path.exists(header+di+'/data_s1.de'))
        assert(os.path.exists(header+di+'/data_s2.en'))
        assert(os.path.exists(header+di+'/data_s2.de'))
        assert(os.path.exists(header+di+'/data_ref.en'))
        assert(os.path.exists(header+di+'/data_ref.de'))
    # create container to hold the infos that will be use to create the xml
    file_list = [] 
    cond_list = []
    File = namedtuple('File', ['name', 'location'])
    Cond = namedtuple('Cond', ['name', 'dev', 'test'])
    # collect the info
    for index, di in enumerate(dirs):
        file_list.append(File(di+'_data_s1', header+di+'/data_s1'))
        file_list.append(File(di+'_data_s2', header+di+'/data_s2'))
        file_list.append(File(di+'_data_ref', header+di+'/data_ref'))
        cond_list.append(Cond('C'+str(index*2), di+'_data_ref', di+'_data_s1'))
        cond_list.append(Cond('C'+str(index*2+1), di+'_data_ref', di+'_data_s2'))
    # create the xml
    doc = xml.dom.minidom.Document()
    root = doc.createElement('system')
    root.setAttribute('command', 'SLURM')
    doc.appendChild(root)
    for item in file_list:
        node = _create_file_node(doc, item)
        root.appendChild(node)
    for item in cond_list:
        node = _create_cond_node(doc, item)
        root.appendChild(node)
    # save the doc
    with open('description.xml', 'w') as fi:
        doc.writexml(fi, indent='\t', addindent='\t', newl='\n', encoding='utf-8')

def _create_file_node(doc, item):
    """
    input:
        doc: xml.dom.minidom.Document object
        item: type of file tuple, namedtuple('name', 'location')

    output:
        null
    """
    filenode = doc.createElement('file')
    namenode = doc.createElement('name')
    namenode.appendChild(doc.createTextNode(item.name))
    typenode = doc.createElement('type')
    typenode.appendChild(doc.createTextNode('parallel'))
    formatnode = doc.createElement('formate')
    formatnode.appendChild(doc.createTextNode('plain'))
    locationnode = doc.createElement('location')
    locationnode.appendChild(doc.createTextNode(item.location))
    sourcenode = doc.createElement('sourceLanguage')
    sourcenode.appendChild(doc.createTextNode('en'))
    targetnode = doc.createElement('targetLanguage')
    targetnode.appendChild(doc.createTextNode('de'))
    filenode.appendChild(namenode)
    filenode.appendChild(typenode)
    filenode.appendChild(locationnode)
    filenode.appendChild(sourcenode)
    filenode.appendChild(targetnode)
    return filenode

def _create_cond_node(doc, item):
    """
    input:
        doc: xml.dom.ninidom.Document object
        item: type of file type, namedtuple('name', 'dev', 'test')

    output:
        null
    """
    confnode = doc.createElement('configuration')
    namenode = doc.createElement('name')
    namenode.appendChild(doc.createTextNode(item.name))
    devdatanode = doc.createElement('devdata')
    devdatanode.appendChild(doc.createTextNode(item.dev))
    testdatanode = doc.createElement('testdata')
    testdatanode.appendChild(doc.createTextNode(item.test))
    prepronode = doc.createElement('preprocessor')
    prepronode.appendChild(doc.createTextNode('bpe'))
    nonode = doc.createElement('noUndo')
    nonode.appendChild(doc.createTextNode('1'))
    decodernode = doc.createElement('decoder')
    dtypenode = doc.createElement('type')
    dtypenode.appendChild(doc.createTextNode('NMT'))
    decodernode.appendChild(dtypenode)
    confnode.appendChild(namenode)
    confnode.appendChild(devdatanode)
    confnode.appendChild(testdatanode)
    confnode.appendChild(prepronode)
    confnode.appendChild(nonode)
    confnode.appendChild(decodernode)
    return confnode



    
