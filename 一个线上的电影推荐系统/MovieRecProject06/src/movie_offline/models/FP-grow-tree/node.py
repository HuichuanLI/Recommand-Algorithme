#-*- coding:utf-8 –*-
__author__ = 'Dodd'

class node:
	def __init__(self,name,parent):
		self.name=name
		self.parent=parent
		self.child=[]
		self.next=None
		self.count=0
		pass
	def findchildnode(self,item):
		for nodes in self.child:
			if nodes.name==item:
				return nodes
			pass
		return None
		pass