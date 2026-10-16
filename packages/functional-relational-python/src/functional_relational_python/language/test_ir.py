import pydantic
import pytest
from hypothesis import given
from hypothesis import strategies as st

from functional_relational_python.compiler import compiler
from functional_relational_python.compiler.test_compiler import (
    Depth,
    models,
    new_model,
    pipelines,
)
from functional_relational_python.domain.entity_graph import FieldName, Optionality
from functional_relational_python.domain.id import Id
from functional_relational_python.language import ir
from functional_relational_python.language.views import View, ViewName


class Text(pydantic.RootModel[str], frozen=True):
    pass


class CompanyId(Id[int]):
    pass


class Company(pydantic.BaseModel):
    id: CompanyId


class DepartmentId(Id[int]):
    pass


class Department(pydantic.BaseModel):
    id: DepartmentId
    company_id: CompanyId


class EmployeeId(Id[int]):
    pass


class Employee(pydantic.BaseModel):
    id: EmployeeId
    department_id: DepartmentId
    mentor_id: EmployeeId | None


def lookup(
    reference: FieldName,
    target: type[pydantic.BaseModel],
    optionality: Optionality = Optionality.REQUIRED,
) -> ir.Lookup:
    id_types: dict[type[pydantic.BaseModel], type[Id[int]]] = {
        Company: CompanyId,
        Department: DepartmentId,
        Employee: EmployeeId,
    }
    return ir.Lookup(
        reference=ir.SourceField(field=reference),
        id_type=id_types[target],
        target=target,
        optionality=optionality,
    )


def own_field(name: FieldName, annotation: ir.RuntimeAnnotation) -> ir.Field:
    model = new_model(ir.SourceName("Model"), {name: annotation})
    match ir.Relation.of(ir.FromSource(source=model)).fields:
        case (ir.Field() as field,):
            return field
        case fields:
            pytest.fail(f"expected one field, got {fields}")


@pytest.mark.parametrize(
    ("annotation", "optionality"),
    [
        (Text, Optionality.REQUIRED),
        (Text | None, Optionality.OPTIONAL),
        (Text | int | None, Optionality.OPTIONAL),
        (list[Text | None], Optionality.REQUIRED),
    ],
    ids=["required", "optional", "optional union", "list of optionals"],
)
def test_a_field_is_optional_iff_its_type_admits_none(
    annotation: ir.RuntimeAnnotation, optionality: Optionality
) -> None:
    field = own_field(FieldName("name"), annotation)

    assert (field.annotation(), field.optionality()) == (annotation, optionality)


@pytest.mark.parametrize(
    ("optionality", "annotation"),
    [
        (Optionality.REQUIRED, Department),
        (Optionality.OPTIONAL, Department | None),
    ],
)
def test_a_looked_up_field_is_typed_by_the_entity_or_none(
    optionality: Optionality, annotation: ir.RuntimeAnnotation
) -> None:
    field = ir.Field.looked_up(
        FieldName("department"),
        lookup(FieldName("department_id"), Department, optionality),
    )

    assert (field.annotation(), field.optionality()) == (annotation, optionality)


def test_a_nested_field_is_as_optional_as_its_lookup() -> None:
    field = ir.NestedField(
        name=FieldName("mentor"),
        value=lookup(FieldName("mentor_id"), Employee, Optionality.OPTIONAL),
        relation=ir.Relation(source=ir.SourceName("Employee"), fields=()),
    )

    assert field.optionality() == Optionality.OPTIONAL


def test_lookups_are_one_per_entity_nested_included_in_first_use_order() -> None:
    department = lookup(FieldName("department_id"), Department)
    mentor = lookup(FieldName("mentor_id"), Employee, Optionality.OPTIONAL)
    company = lookup(FieldName("company_id"), Company)
    relation = ir.Relation(
        source=ir.SourceName("Employee"),
        fields=(
            own_field(FieldName("id"), EmployeeId),
            ir.NestedField(
                name=FieldName("mentor"),
                value=mentor,
                relation=ir.Relation(
                    source=ir.SourceName("Employee"),
                    fields=(
                        ir.NestedField(
                            name=FieldName("department"),
                            value=department,
                            relation=ir.Relation(
                                source=ir.SourceName("Department"),
                                fields=(
                                    ir.Field.looked_up(FieldName("company"), company),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            ir.Field.looked_up(FieldName("department"), department),
        ),
    )

    assert relation.lookups() == (mentor, department, company)


def test_a_relation_of_own_fields_has_no_lookups() -> None:
    assert ir.Relation.of(ir.FromSource(source=Employee)).lookups() == ()


@st.composite
def compiled_views(draw: st.DrawFn) -> compiler.CompiledView:
    """A valid View over a generated Entity, compiled."""
    drawn = draw(models())
    graph = drawn.graph()
    source = draw(st.sampled_from(drawn.entities))
    step = draw(pipelines(graph, source, Depth(2)))
    view = View(name=ViewName.fake(), source=source, step=step)
    return compiler.compile_views(graph, [view]).unwrap().views[0]


def resolved_fields(
    relation: ir.Relation,
) -> tuple[tuple[ir.RelationField, ir.Lookup], ...]:
    """Every field holding a looked-up Entity, nested ones included."""
    found: list[tuple[ir.RelationField, ir.Lookup]] = []
    for field in relation.fields:
        match field:
            case ir.Field(value=ir.Lookup() as value):
                found.append((field, value))
            case ir.NestedField(value=value, relation=nested):
                found.extend(((field, value), *resolved_fields(nested)))
            case ir.Field():
                pass
    return tuple(found)


@given(compiled_views())
def test_a_resolved_field_is_optional_iff_its_lookup_is(
    view: compiler.CompiledView,
) -> None:
    for field, value in resolved_fields(view.relation):
        assert field.optionality() == value.optionality


@given(compiled_views())
def test_lookups_name_each_looked_up_entity_once(
    view: compiler.CompiledView,
) -> None:
    targets = [found.target for found in view.relation.lookups()]

    assert sorted(targets, key=lambda target: target.__name__) == sorted(
        {value.target for _, value in resolved_fields(view.relation)},
        key=lambda target: target.__name__,
    )
