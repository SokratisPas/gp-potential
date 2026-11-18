#pragma once

#include <memory>

#include "primitives.h"
#include "../parsers/xyz-parser.h"


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
	double evaluate(
		double r,
		const std::vector<Neighbor>& neighbors,
		double r_in,
		double r_out
	) const;
	std::string convertToStr() const;	// return the tree as a string representation
	void convertToStrPart(const std::shared_ptr<Node>& node, std::string& out) const;	
	std::shared_ptr<Node> CloneNode(const std::shared_ptr<Node>& node) const;

	private:
	double evaluateNode(
		const std::shared_ptr<Node>& node,
		double r,
		const std::vector<Neighbor>* neighbors,
		double r_in,
		double r_out
	) const;
	std::string nodeTypeToString(NodeType type) const;
	int Size(const std::shared_ptr<Node>& node) const;	
	int Depth(const std::shared_ptr<Node>& node) const;	
	void CollectNodes(
		const std::shared_ptr<Node>& node,
		const std::shared_ptr<Node>& parent,
		size_t childIndex,
		std::vector<NodeLocation>& nodes
	) const;	
	double smoothingFunction(double r_in, double r_out, double r) const;
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
	return evaluateNode(root, r, nullptr, 0.0, 1.0);
}

// ==============================
double Tree::evaluate(
	double r,
	const std::vector<Neighbor>& neighbors,
	double r_in,
	double r_out
) const
{
	return evaluateNode(root, r, &neighbors, r_in, r_out);
}

// ==============================
void Tree::convertToStrPart(const std::shared_ptr<Node>& node, std::string& out) const
{
	// there is no node
	if (!node)
		return;

	// var or const
	switch (node->primitive->type)
	{
	case NodeType::Const:
		out += std::to_string(node->constant);
		return;

	case NodeType::Var:
		out += "r";
		return;

	default:
		break;
	}

	// function
	out += nodeTypeToString(node->primitive->type);
	out += "(";

	for (size_t i = 0; i < node->children.size(); ++i)
	{
		convertToStrPart(node->children[i], out);

		if (i != node->children.size() - 1)
			out += ", ";
	}

	out += ")";
}

// ==============================
std::string Tree::convertToStr() const
{
	std::string out;
	convertToStrPart(root, out);
	return out;
}

// ==============================
double Tree::evaluateNode(
	const std::shared_ptr<Node>& node,
	double r,
	const std::vector<Neighbor>* neighbors,
	double r_in,
	double r_out
) const
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

	if (node->primitive->type == NodeType::SUM_W ||
		node->primitive->type == NodeType::SUM_Mo)
	{
		if (!neighbors)
			throw std::runtime_error("Sum primitive requires neighborhood context.");
		if (node->children.size() != 1)
			throw std::runtime_error("Sum primitive must have exactly one child.");

		const std::string& neighborType =
			node->primitive->type == NodeType::SUM_W ? "W" : "Mo";
		double sum = 0.0;
		for (const auto& neighbor : *neighbors)
		{
			if (neighbor.type == neighborType)
				sum += evaluateNode(
					node->children[0],
					neighbor.distance,
					neighbors,
					r_in,
					r_out
				);
		}
		return sum * smoothingFunction(r_in, r_out, r);
	}

	std::vector<double> childValues;
	childValues.reserve(node->children.size());

	for (const auto& child : node->children)
	{
		childValues.push_back(evaluateNode(child, r, neighbors, r_in, r_out));
	}

	return node->primitive->function(childValues);
}

// ==============================
std::string Tree::nodeTypeToString(NodeType type) const
{
	switch (type)
	{
	case NodeType::Add:   return "Add";
	case NodeType::Sub:   return "Sub";
	case NodeType::Mul:   return "Mul";
	case NodeType::Div:   return "Div";
	case NodeType::Var:   return "r";
	case NodeType::Const: return "Const";
	case NodeType::Pow:   return "Pow";
	case NodeType::SUM_W:   return "SUM_W";
	case NodeType::SUM_Mo:   return "SUM_Mo";
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
	std::vector<NodeLocation>& nodes
) const
{
	if (!node)
		return;

	nodes.push_back({ node, parent, childIndex });

	for (size_t i = 0; i < node->children.size(); ++i)
	{
		CollectNodes(node->children[i], node, i, nodes);
	}
}

// =============================
// smoothing function from:
// "Fast, accurate, and transferable many-body interatomic
// potentials by symbolic regression, Alberto Hernandez, 2019"
// f(r) = (2r^2 - 3r_in^2 + r_out^2) * [(r_out^2 - r^2)^2] / (r_out^2 - r_in^2)^3
double Tree::smoothingFunction(double r_in, double r_out, double r) const
{
		if (r <= r_in)
			return 1.0;
		if (r >= r_out)
			return 0.0;

		const double rSquared = r * r;
		const double rInSquared = r_in * r_in;
		const double rOutSquared = r_out * r_out;
		const double rangeSquared = rOutSquared - rInSquared;

		return (2 * rSquared - 3 * rInSquared + rOutSquared) *
			std::pow(rOutSquared - rSquared, 2) /
			std::pow(rangeSquared, 3);
}