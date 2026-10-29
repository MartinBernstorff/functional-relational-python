import pytest

from functional_relational_python.entity_graph.entity_graph import FieldName
from functional_relational_python.pipeline import ir, steps
from functional_relational_python.pipeline.test_ir import (
    Department,
    Employee,
    lookup,
    of_source,
)


def employee_field(name: FieldName, source: FieldName) -> ir.Field:
    """An Employee's `source` field, named `name`, as a rename would leave it."""
    return ir.Field(
        name=name,
        info=Employee.model_fields[source.root],
        value=ir.SourceField(field=source),
        default=ir.NoDefault(),
    )


@pytest.mark.parametrize(
    "field",
    [
        employee_field(FieldName("id"), FieldName("id")),
        employee_field(FieldName("key"), FieldName("id")),
    ],
    ids=["primary key", "renamed primary key"],
)
def test_the_subjects_id_is_the_primary_key_under_any_name(field: ir.Field) -> None:
    assert steps.is_primary_key(field)


@pytest.mark.parametrize(
    "field",
    [
        employee_field(FieldName("department_id"), FieldName("department_id")),
        employee_field(FieldName("id"), FieldName("department_id")),
        ir.Field.looked_up(
            FieldName("department"),
            lookup(FieldName("department_id"), Department),
            ir.NoDefault(),
        ),
        ir.NestedField(
            name=FieldName("department"),
            value=lookup(FieldName("department_id"), Department),
            relation=of_source(Department),
            default=ir.NoDefault(),
        ),
    ],
    ids=["reference", "reference renamed to id", "looked up", "nested"],
)
def test_a_field_valued_otherwise_is_not_the_primary_key(
    field: ir.RelationField,
) -> None:
    assert not steps.is_primary_key(field)
