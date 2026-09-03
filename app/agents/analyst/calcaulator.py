"""
Analyst Tool 1: Calculator

Performs real arithmetic instead of letting the LLM "calculate" numbers
in its head, which is a common source of silent errors (LLMs are
notoriously unreliable at multi-digit arithmetic and percentage math
"""

import ast
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,  # unary minus, e.g. -5
}

def safe_eval(node):
    if isinstance(node,ast.Constant):
        if isinstance(node.value,(int,float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op_func(safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op_func(safe_eval(node.operand))
    raise ValueError(f"Unsupported AST node type: {type(node)}")

def calculate(expression : str)->float:
    """Safely evaluates a mathematical expression and returns the result."""
    try:
        parsed_expr = ast.parse(expression, mode='eval')
        return safe_eval(parsed_expr.body)
    except (SyntaxError,ValueError,ZeroDivisionError,TypeError) as e:
        raise ValueError(f"Invalid expression: {expression}. Error: {e}")

def average(values : list[float])->float:
    """Calculates the average of a list of numbers."""
    if not values:
        raise ValueError("Cannot calculate average of an empty list.")
    return sum(values) / len(values)

def percentage_change(old_value : float, new_value : float)->float:
    """Calculates the percentage change from old_value to new_value."""
    if old_value == 0:
        raise ValueError("Old value cannot be zero for percentage change calculation.")
    return ((new_value - old_value) / abs(old_value)) * 100

def ratio(a: float,b : float)-> float:
    """Calculates the ratio of a to b."""
    if b == 0:
        raise ValueError("Denominator cannot be zero for ratio calculation.")
    return a / b
