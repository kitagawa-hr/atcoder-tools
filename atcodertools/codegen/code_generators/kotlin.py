from collections import namedtuple
from typing import Dict, Any
import re

from atcodertools.codegen.code_style_config import CodeStyleConfig
from atcodertools.codegen.models.code_gen_args import CodeGenArgs
from atcodertools.codegen.template_engine import render
from atcodertools.fmtprediction.models.format import (
    Pattern,
    SingularPattern,
    ParallelPattern,
    TwoDimensionalPattern,
    Format,
)
from atcodertools.fmtprediction.models.type import Type
from atcodertools.fmtprediction.models.variable import Variable


def _is_formula(s: str) -> bool:
    if re.search(r"\+|-|\*|/", s):
        return True
    return False


def _wrap_formula(s: str) -> str:
    """
    Examples:
        >>> _wrap_formula("N")
        "N"
        >>> _wrap_formula("N + 1")
        "(N + 1)"
    """
    return "({})".format(s) if _is_formula(s) else str(s)


class KotlinCodeGenerator:
    TypeInfo = namedtuple("TypeInfo", ["single", "array", "array_2d"])
    LOOP_VARS = ["i", "j"]
    INPUT_METHOD_MAP = {Type.str: "sc.next()", Type.int: "sc.next().toLong()", Type.float: "sc.next().toDouble()"}
    TYPE_MAP = {
        Type.str: TypeInfo("String", "Array<String>", "Array<Array<String>>"),
        Type.int: TypeInfo("Long", "LongArray", "Array<LongArray>"),
        Type.float: TypeInfo("Double", "DoubleArray", "Array<DoubleArray>"),
    }
    DECL_MAP = {
        Type.str: TypeInfo("String", "StringArray", "Array<Array<String>>"),
        Type.int: TypeInfo("Long", "LongArray", "Array<LongArray>"),
        Type.float: TypeInfo("Double", "DoubleArray", "Array<DoubleArray>"),
    }

    def __init__(self, format_: Format[Variable], config: CodeStyleConfig) -> None:
        self._format = format_
        self._config = config

    def generate_parameters(self) -> Dict[str, Any]:
        return dict(
            formal_arguments=self._formal_arguments(),
            actual_arguments=self._actual_arguments(),
            input_part=self._input_part(),
            prediction_success=True,
        )

    def _input_part(self) -> str:
        paragraphs = []
        for pattern in self._format.sequence:
            paragraphs.append(self.pattern2input_part(pattern))
        return "\n{indent}".format(indent=self._indent(1)).join(paragraphs)

    def pattern2input_part(self, pattern: Pattern) -> str:
        if isinstance(pattern, SingularPattern):
            return self._singular_pattern(pattern)
        elif isinstance(pattern, ParallelPattern):
            return self._parallel_pattern(pattern, self._indent(1))
        elif isinstance(pattern, TwoDimensionalPattern):
            return self._two_dimensional_pattern(pattern, self._indent(1))
        else:
            raise NotImplementedError

    @classmethod
    def _singular_pattern(cls, pattern: SingularPattern) -> str:
        assert isinstance(pattern, SingularPattern)
        var = pattern.all_vars()[0]
        return "val {name} = {input_method}".format(name=var.name, input_method=cls.INPUT_METHOD_MAP[var.type])

    @classmethod
    def _parallel_pattern(cls, pattern: ParallelPattern, indent: str) -> str:
        assert isinstance(pattern, ParallelPattern)
        assert len(pattern.all_vars()) >= 1
        lines = []
        for var in pattern.all_vars():
            declaration = "val {name} = {decl}({size}.toInt())".format(
                name=var.name, decl=cls.DECL_MAP[var.type].array, size=_wrap_formula(str(var.first_index.get_length()))
            )
            lines.append(declaration)

        representative_var = pattern.all_vars()[0]
        lines.append(cls._loop_header(representative_var, False))
        for var in pattern.all_vars():
            lines.append(
                "{indent}{name}[{loop_var}] = {input_method}".format(
                    indent=indent, name=var.name, loop_var=cls.LOOP_VARS[0], input_method=cls.INPUT_METHOD_MAP[var.type]
                )
            )
        lines.append("}")
        return "\n{indent}".format(indent=indent).join(lines)

    @classmethod
    def _two_dimensional_pattern(cls, pattern: TwoDimensionalPattern, indent: str) -> str:
        assert isinstance(pattern, TwoDimensionalPattern)
        assert len(pattern.all_vars()) == 1
        var = pattern.all_vars()[0]
        lines = []
        declaration = "val {name} = {decl1}({size1}.toInt()){{ {decl2}({size2}.toInt()) }}".format(
            name=var.name,
            decl1=cls.DECL_MAP[var.type].array_2d,
            size1=_wrap_formula(str(var.first_index.get_length())),
            decl2=cls.DECL_MAP[var.type].array,
            size2=_wrap_formula(str(var.second_index.get_length())),
        )
        lines.append(declaration)

        first_loop = cls._loop_header(var, False)
        second_loop = cls._loop_header(var, True)
        lines.append(first_loop)
        lines.append(indent + second_loop)
        lines.append(
            "{indent}{indent}{name}[{loop_var1}][{loop_var2}] = {input_method}".format(
                indent=indent,
                name=var.name,
                loop_var1=cls.LOOP_VARS[0],
                loop_var2=cls.LOOP_VARS[1],
                input_method=cls.INPUT_METHOD_MAP[var.type],
            )
        )
        lines.append(indent + "}")
        lines.append("}")
        return "\n{indent}".format(indent=indent).join(lines)

    def _actual_arguments(self) -> str:
        """
            :return the string form of actual arguments e.g. "N, K, a"
        """
        return ", ".join([v.name for v in self._format.all_vars()])

    def _formal_arguments(self) -> str:
        """
            :return the string form of formal arguments e.g. "N: Long, K: Long, a: LongArray"
        """
        return ", ".join(
            [
                "{name}: {type_annotation}".format(
                    type_annotation=self.TYPE_MAP[var.type][var.dim_num()], name=var.name
                )
                for var in self._format.all_vars()
            ]
        )

    def _indent(self, depth: int) -> str:
        return self._config.indent(depth)

    @classmethod
    def _loop_header(cls, var: Variable, for_second_index: bool):
        if for_second_index:
            index = var.second_index
            loop_var = cls.LOOP_VARS[1]
        else:
            index = var.first_index
            loop_var = cls.LOOP_VARS[0]
        assert index is not None
        return "for ({loop_var} in 0 until {length}.toInt()) {{".format(
            loop_var=loop_var, length=_wrap_formula(str(index.get_length()))
        )


def main(args: CodeGenArgs) -> str:
    if args.format is None:
        code_parameters = dict(prediction_success=False)
    else:
        code_parameters = KotlinCodeGenerator(args.format, args.config).generate_parameters()
    return render(
        args.template,
        mod=args.constants.mod,
        yes_str=args.constants.yes_str,
        no_str=args.constants.no_str,
        **code_parameters,
    )
