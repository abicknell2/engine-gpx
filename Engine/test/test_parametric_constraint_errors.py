import pytest

from gpkit import VarKey, ureg

from gpx.dag.parametric import ParametricConstraint, ParametricVariable


def test_parametric_constraint_converts_plain_numbers_to_output_units():
    constraint = ParametricConstraint(constraint_as_list=[229000.0], inputvars={})
    output = ParametricVariable(
        name="Avg Panel Mass",
        varkey=VarKey("Avg Panel Mass", units="lb"),
        unit="lb",
    )
    constraint.update_output_var(output)

    constraint.evaluate()

    assert output.qty.magnitude == pytest.approx(229000.0)
    assert output.qty.units == ureg("lb")


def test_parametric_constraint_allows_dimensionless_output():
    input_var = ParametricVariable(
        name="Scale",
        varkey=VarKey("Scale"),
        qty=2 * ureg.dimensionless,
        magnitude=2,
        is_input=True,
    )
    input_var.update_value()

    constraint = ParametricConstraint(
        constraint_as_list=["Scale", '*', 3.0],
        inputvars={input_var.varkey: input_var},
    )
    output = ParametricVariable(
        name="Dimensionless Result",
        varkey=VarKey("Dimensionless Result"),
    )
    constraint.update_output_var(output)

    constraint.evaluate()

    assert output.qty.magnitude == pytest.approx(6.0)
    assert output.qty.units == ureg.dimensionless
