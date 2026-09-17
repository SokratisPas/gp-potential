#pragma once

#include <memory>

#include "primitives.h"


// --------------------------------------------------------
// Node class, each node has one (inv6, inv12) 
// or two children (example: div, plus)
class Node {
public:
	const Primitive* primitive = nullptr;	// primitive (function, var, const)

	double constant = 0.0;	// usefull only if primitive is constant

	std::vector<std::shared_ptr<Node>> children;

	// in "GeneticProgram::ReplaceInd" we use "std::find" 
	// so we need the == and != operators
	// so we define them for Individual, Tree, Node
	bool operator==(const Node& other) const;
	bool operator!=(const Node& other) const;
};

// --------------------------------------------------------
// struct for parents, children of each node
struct NodeLocation
{
	std::shared_ptr<Node> node;

	std::shared_ptr<Node> parent;

	size_t childIndex;
};

// --------------------------------------------------------
// Tree class
class Tree {
public:
	std::shared_ptr<Node> root;
	
	// we need == and != check explanation in Node
	bool operator==(const Tree& other) const;
	bool operator!=(const Tree& other) const;

	int Size() const;	
	int Depth() const;	
	Tree Clone() const;
	
	std::vector<NodeLocation> CollectNodes() const;
	
	double evaluate(double r) const;
	
	void printTree();	// print tree from root

	void printTreePart(const std::shared_ptr<Node>& node);
	
	std::shared_ptr<Node> CloneNode(const std::shared_ptr<Node>& node) const;

private:
	double evaluateNode(const std::shared_ptr<Node>& node, double r) const;
	
	std::string nodeTypeToString(NodeType type);

	int Size(const std::shared_ptr<Node>& node) const;
	
	int Depth(const std::shared_ptr<Node>& node) const;	
	
	void CollectNodes(
		const std::shared_ptr<Node>& node,
		const std::shared_ptr<Node>& parent,
		size_t childIndex,
		std::vector<NodeLocation>& nodes) const;	
};


// ==============================
bool Node::operator==(const Node& other) const
{
	if (primitive != other.primitive || constant != other.constant || children.size() != other.children.size())
		return false;

	for (size_t i = 0; i < children.size(); ++i)
	{
		if (!children[i] || !other.children[i])
		{
			if (children[i] != other.children[i])
				return false;
			continue;
		}

		if (*children[i] != *other.children[i])
			return false;
	}

	return true;
}

// ==============================
bool Node::operator!=(const Node& other) const
{
	return !(*this == other);
}

// ==============================
bool Tree::operator==(const Tree& other) const
{
	if (!root || !other.root)
		return root == other.root;

	return *root == *other.root;
}

// ==============================
bool Tree::operator!=(const Tree& other) const
{
	return !(*this == other);
}

// ==============================
int Tree::Size() const
{
	return Size(root);
}

// ==============================
int Tree::Depth() const
{
	return Depth(root);
}

// ==============================
Tree Tree::Clone() const
{
	Tree copy;
	copy.root = CloneNode(root);
	return copy;
}

//===============================
std::vector<NodeLocation> Tree::CollectNodes() const
{
	std::vector<NodeLocation> nodes;

	CollectNodes(root, nullptr, 0, nodes);

	return nodes;
}

// ==============================
double Tree::evaluate(double r) const
{
	return evaluateNode(root, r);
}

// ==============================
void Tree::printTreePart(const std::shared_ptr<Node>& node)
{
	// there is no node
	if (!node)
		return;

	// var or const
	switch (node->primitive->type)
	{
	case NodeType::Const:
		std::cout << node->constant;
		return;

	case NodeType::Var:
		std::cout << "r";
		return;

	default:
		break;
	}

	// function
	std::cout << nodeTypeToString(node->primitive->type);
	std::cout << "(";

	for (size_t i = 0; i < node->children.size(); ++i)
	{
		printTreePart(node->children[i]);

		if (i != node->children.size() - 1)
			std::cout << ", ";
	}

	std::cout << ")";
}

// ==============================
void Tree::printTree()
{
	printTreePart(root);
}

// ==============================
double Tree::evaluateNode(const std::shared_ptr<Node>& node, double r) const
{
	if (!node)
		throw std::runtime_error("Null node.");

	switch (node->primitive->type)
	{
	case NodeType::Const:
		return node->constant;

	case NodeType::Var:
		return r;

	default:
		break;
	}

	std::vector<double> childValues;
	childValues.reserve(node->children.size());

	for (const auto& child : node->children)
	{
		childValues.push_back(evaluateNode(child, r));
	}

	return node->primitive->function(childValues);
}

// ==============================
std::string Tree::nodeTypeToString(NodeType type)
{
	switch (type)
	{
	case NodeType::Add:   return "Add";
	case NodeType::Sub:   return "Sub";
	case NodeType::Mul:   return "Mul";
	case NodeType::Div:   return "Div";
	case NodeType::Inv6:  return "Inv6";
	case NodeType::Inv12: return "Inv12";
	case NodeType::Var:   return "r";
	case NodeType::Const: return "Const";
	}

	return "Unknown";
}

// ==============================
int Tree::Size(const std::shared_ptr<Node>& node) const
{
	if (!node)
		return 0;

	int size = 1;

	for (const auto& child : node->children)
		size += Size(child);

	return size;
}

// ==============================
int Tree::Depth(const std::shared_ptr<Node>& node) const
{
	if (!node)
		return 0;

	int maxDepth = 0;

	for (const auto& child : node->children)
		maxDepth = std::max(maxDepth, Depth(child));

	return maxDepth + 1;
}

// ==============================
std::shared_ptr<Node> Tree::CloneNode(const std::shared_ptr<Node>& node) const
{
	if (!node)
		return nullptr;

	auto newNode = std::make_shared<Node>();

	newNode->primitive = node->primitive;
	newNode->constant = node->constant;

	for (const auto& child : node->children)
	{
		newNode->children.push_back(CloneNode(child));
	}

	return newNode;
}

// =============================
void Tree::CollectNodes(
	const std::shared_ptr<Node>& node,
	const std::shared_ptr<Node>& parent,
	size_t childIndex,
	std::vector<NodeLocation>& nodes) const
{
	if (!node)
		return;

	nodes.push_back({ node, parent, childIndex });

	for (size_t i = 0; i < node->children.size(); ++i)
	{
		CollectNodes(node->children[i], node, i, nodes);
	}
}