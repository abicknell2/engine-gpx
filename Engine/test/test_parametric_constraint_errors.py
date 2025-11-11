import pytest

from gpkit import VarKey, ureg

from gpx.dag.parametric import ParametricConstraint, ParametricVariable


def test_parametric_constraint_reports_plain_number_error():
    constraint = ParametricConstraint(constraint_as_list=[229000.0], inputvars={})
    output = ParametricVariable(
        name="Avg Panel Mass",
        varkey=VarKey("Avg Panel Mass", units="lb"),
        unit="lb",
    )
    constraint.update_output_var(output)

    with pytest.raises(ValueError) as excinfo:
        constraint.evaluate()

    message = str(excinfo.value)
    assert "Avg Panel Mass" in message
    assert "plain float" in message
    assert "229000.0" in message
    assert "expected 'lb'" in message


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
    assert str(output.qty.units) == "dimensionless"
