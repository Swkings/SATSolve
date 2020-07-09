#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : SATSolve.py
@Date  : 2020/06/10
@Desc  : 
"""


"""
表达式： ((A|B)>C)>A
后缀表达式： ['A', 'B', '|', 'C', '>', 'A', '>']
主合取范式： (A|B|C) & (A|B|!C) & (A|!B|!C) 
主析取范式： (!A&B&!C) | (A&!B&!C) | (A&!B&C) | (A&B&!C) | (A&B&C) 

Atomic Expression:

      T(A)                   T(B)

     T(A|B)                 F(A|B)
	  /  \                     |
	 /    \                   F(A)
   T(A)   T(B)                 |
                              F(B)
   
     T(!A)                    F(!A)
       |                        |
     F(A)                     T(A)
     
     T(A&B)                  F(A&B)
	   |                      /  \                     
	  T(A)                   /    \                   
       |                   F(A)   F(B)                 
      T(B) 
      
     T(A>B)                 F(A>B)
	  /  \                     |
	 /    \                   T(A)
   F(A)   T(B)                 |
                              F(B)
                              
     T(A=B)                 F(A=B)
	  /  \                   /  \ 
	 /    \                 /    \ 
   F(A)   T(B)            T(A)   F(A)
    |      |               |      |
   F(B)   T(B)            F(B)    T(B)      
"""

import copy
import json
from pyecharts import options as opts
from pyecharts.charts import Tree as pTree

class Node:
	def __init__(self, label=None, value=None):
		self.value = value
		self.label = label
		self.boolValue = ''
		self.call = False
		self.child = []
		self.nextNode = None
		self.parentOp = ''
		self.parentBool = ''

class Tree:
	def __init__(self):
		self.root = None
		self.length = 0
		self._traversePath = list()
		self._path = None

	def __repr__(self):
		print(self.levelOrder(self.root))
		return "<type: %s, size: %d>\n" % (self.__class__.__name__, self.length)

	def __del__(self):
		return

	def addRoot(self, label, value):
		self.root = Node(label, value)
		self.length += 1

	def addChild(self, root, fatherLabel, child):
		if root and root.label == fatherLabel:
			root.child.append(Node(child[0], child[1]))
			self.length += 1
			return True
		elif root:
			for childNode in root.child:
				self.addChild(childNode, fatherLabel, child)
		return False

	def addChildren(self, root: Node, fatherLabel, *args):
		if root and root.label == fatherLabel:
			for label, value in args:
				root.child.append(Node(label, value))
				self.length += 1
			return True
		elif root:
			for childNode in root.child:
				self.addChildren(childNode, fatherLabel, *args)
		return False

	def levelOrder(self, root):
		# 用列表模拟队列，实现广度搜索
		queue = []
		queue.append(root)
		while len(queue) > 0:
			popNode = queue.pop(0)
			self._traversePath.append(popNode)
			for item in popNode.child:
				queue.append(item)
		self.length = len(self._traversePath)

		self._path = [{item.label: item.value} for item in self._traversePath]
		return self._path

	def Tree2Dict(self, root):
		node = {}
		if len(root.child) > 0:
			node['name'] = root.boolValue + '( ' + root.label + ' )' + ' op: ' + root.value
			node['children'] = []
			for child in root.child:
				node['children'].append(self.Tree2Dict(child))
		else:
			# return {'name': root.label, 'children': [{'name': '*'}]}
			return {'name': root.boolValue + '( ' + root.label + ' )' + ' op: ' + root.value}
		return node

	def levelOrder_lv(self, root):
		# 用列表模拟队列，实现广度搜索
		queue = []
		queue.append(root)
		level = 0
		while len(queue) > 0:
			popNode = queue.pop(0)
			popNode.value = level
			self._traversePath.append(popNode)
			for item in popNode.child:
				item.value = level + 1
				queue.append(item)
			level += 1
		self._path = [{item.label: item.value} for item in self._traversePath]
		return self._path

	def DFS(self, root):
		LinkCollect = []
		LinkList = []
		Link = []
		queue = []
		queue.append(root)
		LinkList.append(Link)
		while len(queue) > 0:
			popNode = queue.pop()
			LinkList[-1].append([popNode.label, popNode.boolValue])
			if len(popNode.child) == 0:
				LinkCollect.append(copy.deepcopy(LinkList.pop()))
			if len(popNode.child) > 1:
				LL = LinkList.pop()
				for item in range(len(popNode.child)):
					LinkList.append(copy.deepcopy(LL))
			for item in popNode.child:
				queue.append(item)
		return LinkCollect

def Expression2Words(Expression):
	"""
	中序表达式 转 后序表达式
	:param Expression: 中序表达式字符串
	:return:
	"""
	Expression = [item for item in Expression[::-1]]
	operatorStack = []
	wordStack = []
	words = {}
	while len(Expression) != 0:
		word = Expression.pop()
		if word == '(':
			operatorStack.append(word)
		elif word in ['|', '&', '!', '>', '=']:
			isLeftParen = False
			isLowerOperator = False
			isEmpty = len(operatorStack) == 0
			if not isEmpty:  # 非空
				stackTop = operatorStack[-1]
				isLeftParen = stackTop == '('
				if stackTop != '!' and word == '!':
					isLowerOperator = True
			# if not (isEmpty or isLeftParen or isLowerOperator):
			# if not isEmpty and not isLeftParen and not isLowerOperator:
			# 	print(operatorStack)
			while not isEmpty and not isLeftParen and not isLowerOperator:
				wordStack.append(operatorStack.pop())
				isEmpty = len(operatorStack) == 0
				if not isEmpty:  # 非空
					stackTop = operatorStack[-1]
					isLeftParen = stackTop == '('
					if stackTop != '!' and word == '!':
						isLowerOperator = True

			operatorStack.append(word)
		elif word == ')':
			oWord = operatorStack.pop()
			while oWord != '(':
				wordStack.append(oWord)
				if len(operatorStack) == 0:
					break
				oWord = operatorStack.pop()
		else:
			wordStack.append(word)
			words[word] = word
	while len(operatorStack) > 0:
		wordStack.append(operatorStack.pop())
	return wordStack, sorted(words)

def SATTree(wordStack):
	"""
	生成SAT分解节点，并构成树结构
	:param wordStack: 后序表达式
	:return:
	"""
	wordStackCopy = wordStack[::-1]
	nodeList = []
	level = 0
	while len(wordStackCopy) > 0:
		word = wordStackCopy.pop()
		if word == '!':
			leftNode = nodeList.pop()
			leftValueExpr = leftNode.label
			valueExpr = '!(' + leftValueExpr + ')'
			label = valueExpr
			node = Node(label=label, value=word)
			leftNode.parentOp = word
			node.child.append(leftNode)
			nodeList.append(node)
		elif word in ['|', '&', '>', '=']:
			rightNode = nodeList.pop()
			rightValueExpr = rightNode.label
			leftNode = nodeList.pop()
			leftValueExpr = leftNode.label
			valueExpr = '(' + leftValueExpr + word + rightValueExpr + ')'

			label = valueExpr
			node = Node(label=label, value=word)
			leftNode.parentOp = word
			rightNode.parentOp = word
			node.child.append(leftNode)
			node.child.append(rightNode)
			nodeList.append(node)
			level += 1
		else:
			label = word
			node = Node(label=label, value='#')
			nodeList.append(node)
	return nodeList

def SAT(root):
	"""
	SAT推演，递归扩展
	主要思路：
		1、从根节点开始，根据规则构成子节点
		2、通过每一个节点的nextNode记录该节点所属链路的子树
		3、根据步骤1中构成子节点的规则，对应的将父节点的子树扩展到子节点中
		4、递归，对每个子节点执行1、2、3
		5、判断子节点是否赋值（T or F） 并且为叶子节点（op=‘#’），若是，将该节点的子树链接到该点的子节点，执行4，否则返回
	:param root: SATTree
	:return:
	"""
	if root.call is False and len(root.child) >= 2:
		root.call = True
		A = root.child[0]
		B = root.child[1]
		op = A.parentOp
		boolOP = A.parentBool
		if boolOP == '':
			boolOP = root.boolValue
		if boolOP == 'T':
			if op == '|':
				"""
				'|'运算，生成2个子节点，子节点扩展子树
				"""
				A.boolValue = 'T'
				B.boolValue = 'T'
				A.nextNode = root.nextNode
				B.nextNode = root.nextNode
				root.child = []
				root.child.append(A)
				root.child.append(B)
			elif op == '&':
				"""
				'&'运算，生成1个子节点，将另外一个字节点设置为该子节点的子树，（因为是同一条链路）
				"""
				A.boolValue = 'T'
				B.boolValue = 'T'
				for c in A.child:
					c.parentBool = A.boolValue
				B.nextNode = root.nextNode
				A.nextNode = B
				root.child = []
				root.child.append(A)
			elif op == '>':
				"""
				'>' 同 '|'
				"""
				A.boolValue = 'F'
				B.boolValue = 'T'
				A.nextNode = root.nextNode
				B.nextNode = root.nextNode
				root.child = []
				root.child.append(A)
				root.child.append(B)
			elif op == '=':
				"""
				'='运算，生成两个2节点子树，在每个子树的末节点链接父节点的子树
				"""
				As = copy.deepcopy(A)
				Bs = copy.deepcopy(B)
				A.boolValue = 'F'
				As.boolValue = 'T'
				B.boolValue = 'T'
				Bs.boolValue = 'F'

				Bs.nextNode = root.nextNode
				A.nextNode = Bs

				B.nextNode = root.nextNode
				As.nextNode = B
				root.child = []
				root.child.append(A)
				root.child.append(As)
		elif boolOP == 'F':  # 同boolOP == 'T'，根据规则
			if op == '|':
				A.boolValue = 'F'
				B.boolValue = 'F'
				for c in A.child:
					c.parentBool = A.boolValue
				B.nextNode = root.nextNode
				A.nextNode = B
				root.child = []
				root.child.append(A)
			elif op == '&':
				A.boolValue = 'F'
				B.boolValue = 'F'
				A.nextNode = root.nextNode
				B.nextNode = root.nextNode
				root.child = []
				root.child.append(A)
				root.child.append(B)
			elif op == '>':
				A.boolValue = 'T'
				B.boolValue = 'F'
				# 更换节点
				for c in A.child:
					c.parentBool = A.boolValue
				B.nextNode = root.nextNode
				A.nextNode = B
				root.child = []
				root.child.append(A)
			elif op == '=':
				As = copy.deepcopy(A)
				Bs = copy.deepcopy(B)
				A.boolValue = 'T'
				As.boolValue = 'F'
				B.boolValue = 'T'
				Bs.boolValue = 'F'
				Bs.nextNode = root.nextNode
				A.nextNode = Bs
				B.nextNode = root.nextNode
				As.nextNode = B
				root.child = []
				root.child.append(A)
				root.child.append(As)
	elif root.call is False and root.value == '!':  # '!'优先级最高，单独处理
		root.call = True
		A = root.child[0]
		boolOP = root.boolValue
		A.nextNode = root.nextNode
		if boolOP == 'T':
			A.boolValue = 'F'
		elif boolOP == 'F':
			A.boolValue = 'T'
	elif root.call is False and root.value == '#' and root.nextNode is not None:
		"""
		搜索到叶子节点，并且有子树
		"""
		root.call = True
		root.child = [root.nextNode]
		root.nextNode = None
	for i in range(len(root.child)):
		SAT(root.child[i])
	return root


if __name__ == '__main__':
	# expression1 = 'A>(B|C)'
	# expression2 = '((A|B)>C)>A'
	# expression3 = '((p>q)>p)>p'
	OP = {
		'|': ' or ',
		'&': ' and ',
		'>': ' imply ',
		'=': ' equiv ',
		'!': 'not'
	}
	print("\n===========Algorithm Start===========\n")
	while 1:
		print("[ 或(or)：|, 与(and)：&, 非(not)：!, 蕴含(imply)：>, 等价(equiv)：= ]")
		s = input('Command >> ')
		s = s.strip()
		expression = s.replace(' ', '')
		if expression == 'break' or expression == 'stop' or expression == 'exit' or expression == '-1':
			break
		if expression == '':
			continue
		boolValue = input('Command >> 证永真T，永假F: ').strip()
		if boolValue != 'T' or boolValue != 'F':
			boolValue = 'F'
		print('表达式：', expression)
		wordStack, words = Expression2Words(expression)
		print('后缀表达式：', wordStack)
		print('SAT Expr：', boolValue + '(', expression, ')')

		# 构造STA树
		nodeList = SATTree(wordStack)

		T = Tree()
		T.root = nodeList[-1]
		# 证永真T，永假F
		# T.root.boolValue = 'T'
		# T.root.boolValue = 'F'
		T.root.boolValue = boolValue
		root = SAT(T.root)
		T.root = root
		LinkList = T.DFS(T.root)
		TDict = T.Tree2Dict(root)
		print('推演树状结构:\n', json.dumps(TDict, indent=4, separators=(', ', ': ')))
		print("\n")
		# tree = (
		# 		pTree()
		# 		.add("",
		# 		     [TDict],
		# 		     orient="TB",
		# 		     initial_tree_depth=-1,
		# 		     pos_top='15%',
		# 		     pos_bottom='5%',
		# 		     pos_left='20%',
		# 	         pos_right='5%')
		# 		.set_global_opts(title_opts=opts.TitleOpts(title="SAT Solve Tree")))
		#
		# exprList = [item for item in expression]
		# for it in range(len(exprList)):
		# 	if exprList[it] in ['!', '|', '&', '>', '=']:
		# 		exprList[it] = OP[exprList[it]]
		# saveFileName = './output/E[ ' + ''.join(exprList) + ' ].html'
		# tree.render(saveFileName)

		# 判断是否可满足
		boolCount = {}
		for word in words:
			boolCount[word] = {'T': 0, 'F': 0}
		satisfiable = []
		for link in LinkList:
			BC = copy.deepcopy(boolCount)
			for expr, boolValue in link:
				if len(expr) == 1:
					BC[expr][boolValue] += 1
			satisfiable.append(BC)
		for it in range(len(satisfiable)):
			BC = satisfiable[it]
			isSatisfiable = True
			for key in BC.keys():
				if BC[key]['T'] > 0 and BC[key]['F'] > 0:
					satisfiable[it]['Prove'] = 'NonSatisfiable'
					isSatisfiable = False
					break
			if isSatisfiable:
				satisfiable[it]['Prove'] = 'Satisfiable'
		print('Prove Result:\n', json.dumps(satisfiable, indent=4, separators=(', ', ': ')), '\n')
