from lxml import etree
import numpy as np
import xml.etree.ElementTree as ET
import glob, os

def crearXML(det,directoy):
# crear xml a partir de detecciones

	numObjects = np.shape(det)[0]

	#print(numObjects)
	# Create the root element
	page = etree.Element('annotation')

	# Make a new document tree
	doc = etree.ElementTree(page)

	# Add the subelements

	for i in range(0,numObjects):


		name = str(det[i][0])
		#print(name)
		
		pageElement = etree.SubElement(page, 'Object' )
		pageElementBounding = etree.SubElement(pageElement, 'name')
		pageElementBounding.text = name
		pageElementBounding = etree.SubElement(pageElement, 'bdnbox')

		pageElement = etree.SubElement(pageElementBounding, 'xmin')
		pageElement.text = str(det[i][1])
		pageElement = etree.SubElement(pageElementBounding, 'ymin')
		pageElement.text = str(det[i][2])
		pageElement = etree.SubElement(pageElementBounding, 'xmax')
		pageElement.text = str(det[i][3])
		pageElement = etree.SubElement(pageElementBounding, 'ymax')
		pageElement.text = str(det[i][4])
		

	# Save to XML file
	#outFile = open('/home/marc/Dropbox/tfmDeepLearning/semana4/boundingBox/files/voc0001.xml', 'wb')
	outFile = open(directoy, 'wb')
	
	#doc.write(outFile, xml_declaration=True, encoding='utf-16')
	doc.write(outFile)
	#print('doneInter')

def crearXML2(det,directoy):
# crear xml a partir de detecciones incluye confianza

	numObjects = np.shape(det)[0]

	#print(numObjects)
	# Create the root element
	page = etree.Element('annotation')

	# Make a new document tree
	doc = etree.ElementTree(page)

	# Add the subelements

	for i in range(0,numObjects):


		name = str(det[i][0])
		#print(name)
		
		pageElement = etree.SubElement(page, 'Object' )
		pageElementBounding = etree.SubElement(pageElement, 'name')
		pageElementBounding.text = name
		pageElementBounding = etree.SubElement(pageElement, 'bdnbox')

		pageElement = etree.SubElement(pageElementBounding, 'confidence')
		pageElement.text = str(det[i][1])
		pageElement = etree.SubElement(pageElementBounding, 'xmin')
		pageElement.text = str(det[i][2])
		pageElement = etree.SubElement(pageElementBounding, 'ymin')
		pageElement.text = str(det[i][3])
		pageElement = etree.SubElement(pageElementBounding, 'xmax')
		pageElement.text = str(det[i][4])
		pageElement = etree.SubElement(pageElementBounding, 'ymax')
		pageElement.text = str(det[i][5])
		

	# Save to XML file
	#outFile = open('/home/marc/Dropbox/tfmDeepLearning/semana4/boundingBox/files/voc0001.xml', 'wb')
	outFile = open(directoy, 'wb')
	
	#doc.write(outFile, xml_declaration=True, encoding='utf-16')
	doc.write(outFile)
	#print('doneInter')
	
def leerXMLpropio(directorio):
# leer XML propio, lee todo el directorio
	#tree = ET.parse('/home/marc/Dropbox/tfmDeepLearning/semana4/boundingBox/output.xml')
	tree = ET.parse(directorio)
	
	root = tree.getroot()
	matriz = []

	for country in root.findall('Object'):

		
	  	name = country.find('name').text
	  	#confidence = country.find('bdnbox').find('confidence').text
	  	xmin = country.find('bdnbox').find('xmin').text
	  	ymin = country.find('bdnbox').find('ymin').text
	  	xmax = country.find('bdnbox').find('xmax').text
	  	ymax = country.find('bdnbox').find('ymax').text
	  	#matriz.append([name,confidence,xmin,ymin,xmax,ymax])
	  	matriz.append([name,xmin,ymin,xmax,ymax])
	  	#print (xmin,ymin,xmax,ymax, name)
	return matriz


def leerXMLpropioClase(directorio,clase):
# leer xml propio, lee la clase seleccioanda
	#tree = ET.parse('/home/marc/Dropbox/tfmDeepLearning/semana4/boundingBox/output.xml')
	tree = ET.parse(directorio)
	
	root = tree.getroot()
	matriz = []

	for country in root.findall('Object'):

		name = country.find('name').text
		confidence = country.find('bdnbox').find('confidence').text

		if (name==clase):
		#if (name == clase)and(float(confidence)>0.1):

			confidence = country.find('bdnbox').find('confidence').text
			xmin = country.find('bdnbox').find('xmin').text
			ymin = country.find('bdnbox').find('ymin').text
			xmax = country.find('bdnbox').find('xmax').text
			ymax = country.find('bdnbox').find('ymax').text
			
			matriz.append([name,confidence,xmin,ymin,xmax,ymax])
			#print(name)
	return matriz

def leerXMLimageLabel(directorio):
# Para leer todos los xml del directorio

	#os.chdir("/home/marc/Dropbox/tfmDeepLearning/semana4/anotar2")
	os.chdir(directorio)
	matriz = []
	for file in glob.glob("*.xml"):

		
		tree = ET.parse(file)
		root = tree.getroot()

		for country in root.findall('object'):
			name = country.find('name').text
			xmin = country.find('bndbox').find('xmin').text
			ymin = country.find('bndbox').find('ymin').text
			xmax = country.find('bndbox').find('xmax').text
			ymax = country.find('bndbox').find('ymax').text
			#print (xmin,ymin,xmax,ymax, name)
			matriz.append([name,xmin,ymin,xmax,ymax])

	#print(matriz[3])
	return matriz
def leerXMLimageLabel2(directorio):
# Para leer solo el xml seleccionado

	#os.chdir("/home/marc/Dropbox/tfmDeepLearning/semana4/anotar2")
	matriz = []
		
	tree = ET.parse(directorio)
	root = tree.getroot()

	for country in root.findall('object'):
		name = country.find('name').text
		xmin = country.find('bndbox').find('xmin').text
		ymin = country.find('bndbox').find('ymin').text
		xmax = country.find('bndbox').find('xmax').text
		ymax = country.find('bndbox').find('ymax').text
		#print (xmin,ymin,xmax,ymax, name)
		matriz.append([name,xmin,ymin,xmax,ymax])

	#print(matriz[3])
	return matriz


def leerXMLimageLabel2clase(directorio,clase):
# Para leer solo el xml seleccionado

	#os.chdir("/home/marc/Dropbox/tfmDeepLearning/semana4/anotar2")
	matriz = []
	
	tree = ET.parse(directorio)
	root = tree.getroot()

	for country in root.findall('object'):


		name = country.find('name').text
		difficulty = country.find('difficult').text
		#confidence = country.find('bndbox').find('confidenc').text
		#print(confidence)
		if (name != clase) | (difficulty == '1'):
		#if (name != clase):
			continue

		#name = country.find('name').text
		#difficulty = country.find('difficult').text
		xmin = country.find('bndbox').find('xmin').text
		ymin = country.find('bndbox').find('ymin').text
		xmax = country.find('bndbox').find('xmax').text
		ymax = country.find('bndbox').find('ymax').text
		#print (xmin,ymin,xmax,ymax, name)
		matriz.append([name,xmin,ymin,xmax,ymax])
		
	#print(matriz[3])
	return matriz
