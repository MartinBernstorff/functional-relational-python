import itertools
from collections.abc import Sequence
from uuid import UUID

import pytest
from pydantic import BaseModel, RootModel
from safe_result import Err, Ok, Result

from functional_relational_python.entity_graph import entity_graph
from functional_relational_python.entity_graph.id import Id, IdBase
from functional_relational_python.support.diagnostic import DiagnosticsError


class DepartmentId(Id[int]):
    pass


class EmployeeId(Id[int]):
    pass


class ProjectId(Id[int]):
    pass


class Department(BaseModel):
    id: DepartmentId


class Employee(BaseModel):
    id: EmployeeId
    department_id: DepartmentId
    mentor: EmployeeId | None


class Project(BaseModel):
    id: ProjectId
    owner_id: EmployeeId
    department_id: DepartmentId | None


class OtherDepartment(BaseModel):
    id: DepartmentId


class ThirdDepartment(BaseModel):
    id: DepartmentId


class AssignmentId(Id[int]):
    pass


class BudgetId(Id[str]):
    pass


class Name(RootModel[str]):
    pass


class Assignment(BaseModel):
    id: AssignmentId
    project_id: ProjectId


class Budget(BaseModel):
    id: BudgetId
    department_id: DepartmentId


class Unidentified(BaseModel):
    name: Name


class NonIdPrimaryKey(BaseModel):
    id: Name


class OptionalId(BaseModel):
    id: DepartmentId | None


class AppId(RootModel[UUID], frozen=True):
    """An ID base an application defines, rather than subclassing frp's `Id`."""


class TeamId(AppId):
    pass


class MemberId(AppId):
    pass


class Team(BaseModel):
    id: TeamId


class Member(BaseModel):
    id: MemberId
    team_id: TeamId
    department_id: DepartmentId


def _diagnostics(
    entities: Sequence[type[BaseModel]],
) -> frozenset[entity_graph.EntityGraphDiagnostic]:
    match entity_graph.EntityGraph.build(entities, IdBase.fake()):
        case Ok(graph):
            pytest.fail(f"Expected diagnostics, got {graph}")
        case Err(error):
            return frozenset(error.diagnostics)


def test_references_resolve_to_the_entity_owning_their_id_type() -> None:
    result = entity_graph.EntityGraph.build(
        [Department, Employee, Project], IdBase.fake()
    )

    assert isinstance(result, Ok)
    assert result.value.entities[Project].references == (
        entity_graph.Reference(
            field=entity_graph.FieldName("owner_id"),
            id_type=EmployeeId,
            target=Employee,
            optionality=entity_graph.Optionality.REQUIRED,
        ),
        entity_graph.Reference(
            field=entity_graph.FieldName("department_id"),
            id_type=DepartmentId,
            target=Department,
            optionality=entity_graph.Optionality.OPTIONAL,
        ),
    )


def test_an_entity_may_reference_itself() -> None:
    result = entity_graph.EntityGraph.build([Department, Employee], IdBase.fake())

    assert isinstance(result, Ok)
    assert [
        reference.target for reference in result.value.entities[Employee].references
    ] == [Department, Employee]


def test_entities_record_their_primary_key_type() -> None:
    result = entity_graph.EntityGraph.build([Department], IdBase.fake())

    assert isinstance(result, Ok)
    assert result.value.entities[Department].id_type is DepartmentId


@pytest.mark.parametrize(
    ("entities", "expected"),
    [
        pytest.param(
            [Department, OtherDepartment],
            entity_graph.DuplicatePrimaryKey(
                id_type=DepartmentId,
                entities=frozenset({Department, OtherDepartment}),
            ),
            id="two entities",
        ),
        pytest.param(
            [Department, OtherDepartment, ThirdDepartment],
            entity_graph.DuplicatePrimaryKey(
                id_type=DepartmentId,
                entities=frozenset({Department, OtherDepartment, ThirdDepartment}),
            ),
            id="three entities",
        ),
    ],
)
def test_an_id_type_keying_several_entities_is_a_diagnostic(
    entities: Sequence[type[BaseModel]],
    expected: entity_graph.DuplicatePrimaryKey,
) -> None:
    assert _diagnostics(entities) == frozenset({expected})


@pytest.mark.parametrize(
    ("entities", "expected"),
    [
        pytest.param(
            [Employee],
            entity_graph.UnresolvedReference(
                entity=Employee,
                field=entity_graph.FieldName("department_id"),
                id_type=DepartmentId,
            ),
            id="required",
        ),
        pytest.param(
            [Department, Project],
            entity_graph.UnresolvedReference(
                entity=Project,
                field=entity_graph.FieldName("owner_id"),
                id_type=EmployeeId,
            ),
            id="required, beside a resolved reference",
        ),
        pytest.param(
            [Employee, Project],
            entity_graph.UnresolvedReference(
                entity=Project,
                field=entity_graph.FieldName("department_id"),
                id_type=DepartmentId,
            ),
            id="optional",
        ),
    ],
)
def test_a_reference_without_an_entity_is_a_diagnostic(
    entities: Sequence[type[BaseModel]],
    expected: entity_graph.UnresolvedReference,
) -> None:
    assert expected in _diagnostics(entities)


@pytest.mark.parametrize(
    "model",
    [Unidentified, NonIdPrimaryKey, OptionalId],
)
def test_a_model_without_an_id_primary_key_is_a_diagnostic(
    model: type[BaseModel],
) -> None:
    assert _diagnostics([model]) == frozenset(
        {entity_graph.NotAnEntity(model=model, id_base=IdBase.fake())}
    )


def test_every_diagnostic_is_reported_at_once() -> None:
    assert _diagnostics([Department, OtherDepartment, Unidentified, Assignment]) == {
        entity_graph.DuplicatePrimaryKey(
            id_type=DepartmentId,
            entities=frozenset({Department, OtherDepartment}),
        ),
        entity_graph.NotAnEntity(model=Unidentified, id_base=IdBase.fake()),
        entity_graph.UnresolvedReference(
            entity=Assignment,
            field=entity_graph.FieldName("project_id"),
            id_type=ProjectId,
        ),
    }


def test_a_reference_to_a_duplicated_id_type_is_not_also_unresolved() -> None:
    diagnostics = _diagnostics([Department, OtherDepartment, Budget])

    assert diagnostics == {
        entity_graph.DuplicatePrimaryKey(
            id_type=DepartmentId,
            entities=frozenset({Department, OtherDepartment}),
        ),
    }


def test_the_error_message_names_the_offending_types() -> None:
    match entity_graph.EntityGraph.build([Employee], IdBase.fake()):
        case Ok(graph):
            pytest.fail(f"Expected diagnostics, got {graph}")
        case Err(error):
            assert "Employee.department_id" in str(error)
            assert "DepartmentId" in str(error)


@pytest.mark.parametrize(
    "entities",
    [
        pytest.param([Department, Employee, Project, Assignment], id="valid"),
        pytest.param(
            [Department, OtherDepartment, Employee, Unidentified, Assignment],
            id="invalid",
        ),
    ],
)
def test_reordering_the_entities_gives_the_same_result(
    entities: Sequence[type[BaseModel]],
) -> None:
    first, *rest = [
        _comparable(entity_graph.EntityGraph.build(permutation, IdBase.fake()))
        for permutation in itertools.permutations(entities)
    ]

    assert all(other == first for other in rest)


def _comparable(
    result: Result[
        entity_graph.EntityGraph, DiagnosticsError[entity_graph.EntityGraphDiagnostic]
    ],
) -> entity_graph.EntityGraph | frozenset[entity_graph.EntityGraphDiagnostic]:
    match result:
        case Ok(graph):
            return graph
        case Err(error):
            return frozenset(error.diagnostics)


class Label(RootModel[str]):
    pass


@pytest.mark.parametrize(
    ("annotation", "split"),
    [
        (Label, (entity_graph.Optionality.REQUIRED, Label)),
        (Label | None, (entity_graph.Optionality.OPTIONAL, Label)),
        (Label | int | None, (entity_graph.Optionality.OPTIONAL, Label | int)),
        (list[Label | None], (entity_graph.Optionality.REQUIRED, list[Label | None])),
    ],
    ids=["required", "optional", "optional union", "list of optionals"],
)
def test_an_annotation_is_optional_iff_it_admits_none(
    annotation: entity_graph.RuntimeAnnotation,
    split: tuple[entity_graph.Optionality, entity_graph.RuntimeAnnotation],
) -> None:
    optionality, required = split

    assert entity_graph.Optionality.of_annotation(
        annotation
    ) == entity_graph.SplitAnnotation(optionality=optionality, required=required)


@pytest.mark.parametrize("optionality", list(entity_graph.Optionality))
def test_an_annotation_built_with_an_optionality_is_recognised_as_it(
    optionality: entity_graph.Optionality,
) -> None:
    assert entity_graph.Optionality.of_annotation(
        optionality.annotate(Label)
    ) == entity_graph.SplitAnnotation(optionality=optionality, required=Label)


@pytest.mark.parametrize(
    ("optionality", "text"),
    [
        (entity_graph.Optionality.REQUIRED, "Label"),
        (entity_graph.Optionality.OPTIONAL, "Label | None"),
    ],
)
def test_an_annotation_is_written_with_its_optionality(
    optionality: entity_graph.Optionality, text: str
) -> None:
    assert optionality.annotate_text(Label("Label")) == Label(text)


def test_the_id_base_decides_which_fields_are_references() -> None:
    graph = entity_graph.EntityGraph.build([Team, Member], IdBase(AppId)).unwrap()

    assert graph.entities[Member].references == (
        entity_graph.Reference(
            field=entity_graph.FieldName("team_id"),
            id_type=TeamId,
            target=Team,
            optionality=entity_graph.Optionality.REQUIRED,
        ),
    )


def test_a_model_keyed_by_an_id_of_another_base_is_not_an_entity() -> None:
    match entity_graph.EntityGraph.build([Department], IdBase(AppId)):
        case Ok(graph):
            pytest.fail(f"Expected diagnostics, got {graph}")
        case Err(error):
            assert error.diagnostics == (
                entity_graph.NotAnEntity(model=Department, id_base=IdBase(AppId)),
            )
            assert "AppId" in str(error)
