#pragma once

#include <vector>
#include <string>
#include <functional>
#include <cmath>

// --------------------------------------------------------
// types of Node
enum class NodeType {
	Add,
	Sub,
	Div,
	Mul,
	Inv6,
	Inv12,
	Const,
	Var
};

// --------------------------------------------------------
// Primitive : function, ephemeral constant, terminal (leaf)
struct Primitive {
	NodeType type;

	int arity;	// number of inputs of primitive

	std::function<double(const std::vector<double>&)> function;
};

// --------------------------------------------------------
// --------------------------------------------------------
// Define Primitives
Primitive Add{
	NodeType::Add,
	2,
	[](const std::vector<double>& x)
	{
		return x[0] + x[1];
	}
};
// ==================
Primitive Sub{
	NodeType::Sub,
	2,
	[](const std::vector<double>& x)
	{
		return x[0] - x[1];
	}
};
// ==================
Primitive Div{
	NodeType::Div,
	2,
	[](const std::vector<double>& x)
	{
		if (std::abs(x[1]) < 1e-6)	// catch div with 0
			return 1e6;
		else
			return x[0] / x[1];
	}
};
// ==================
Primitive Mul{
	NodeType::Mul,
	2,
	[](const std::vector<double>& x)
	{
		return x[0] * x[1];
	}
};
// ==================
Primitive Inv6{
	NodeType::Inv6,
	1,
	[](const std::vector<double>& x)
	{
		if (std::abs(x[0]) < 1e-6)
			return 1e6;
		else
			return 1.0 / std::pow(x[0], 6);
	}
};
// ==================
Primitive Inv12{
	NodeType::Inv12,
	1,
	[](const std::vector<double>& x)
	{
		if (std::abs(x[0]) < 1e-6)
			return 1e6;
		else
			return 1.0 / std::pow(x[0], 12);
	}
};
// ==================
Primitive Var{
	NodeType::Var,
	0,	// var has no children
	nullptr
};
// ==================
Primitive Const{
	NodeType::Const,
	0,	// constant has no children
	nullptr
};