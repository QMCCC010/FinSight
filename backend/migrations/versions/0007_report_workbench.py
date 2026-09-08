"""Upgrade generated reports into an editable asynchronous workbench.

Revision ID: 0007_report_workbench
Revises: 0006_chat_background_refresh
"""

from alembic import op
import sqlalchemy as sa


revision = "0007_report_workbench"
down_revision = "0006_chat_background_refresh"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # MySQL auto-commits DDL. Keep the migration recoverable when an earlier
    # statement succeeded but a later one failed during a development restart.
    inspector = sa.inspect(op.get_bind())
    columns = {item["name"] for item in inspector.get_columns("generated_reports")}

    def add_if_missing(name: str, column: sa.Column) -> None:
        if name not in columns:
            op.add_column("generated_reports", column)
            columns.add(name)

    add_if_missing("status", sa.Column("status", sa.String(length=20), nullable=False, server_default="COMPLETED"))
    add_if_missing("progress", sa.Column("progress", sa.Integer(), nullable=False, server_default="100"))
    add_if_missing("status_text", sa.Column("status_text", sa.String(length=255), nullable=True))
    add_if_missing("error", sa.Column("error", sa.Text(), nullable=True))
    if "source_types" not in columns:
        op.add_column("generated_reports", sa.Column("source_types", sa.JSON(), nullable=True))
        op.execute(sa.text("UPDATE generated_reports SET source_types = '[]' WHERE source_types IS NULL"))
        op.alter_column("generated_reports", "source_types", existing_type=sa.JSON(), nullable=False)
        columns.add("source_types")
    if "selected_document_ids" not in columns:
        op.add_column("generated_reports", sa.Column("selected_document_ids", sa.JSON(), nullable=True))
        op.execute(sa.text("UPDATE generated_reports SET selected_document_ids = '[]' WHERE selected_document_ids IS NULL"))
        op.alter_column("generated_reports", "selected_document_ids", existing_type=sa.JSON(), nullable=False)
        columns.add("selected_document_ids")
    add_if_missing("completed_at", sa.Column("completed_at", sa.DateTime(), nullable=True))
    add_if_missing("is_deleted", sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.false()))

    inspector = sa.inspect(op.get_bind())
    indexes = {item["name"] for item in inspector.get_indexes("generated_reports")}
    if "ix_generated_reports_status" not in indexes:
        op.create_index("ix_generated_reports_status", "generated_reports", ["status"], unique=False)
    if "ix_generated_reports_is_deleted" not in indexes:
        op.create_index("ix_generated_reports_is_deleted", "generated_reports", ["is_deleted"], unique=False)
    if "report_versions" not in inspector.get_table_names():
        op.create_table(
            "report_versions",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("report_id", sa.Integer(), nullable=False),
            sa.Column("version_number", sa.Integer(), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=False),
            sa.Column("content_markdown", sa.Text(), nullable=False),
            sa.Column("citations", sa.JSON(), nullable=False),
            sa.Column("change_type", sa.String(length=32), nullable=False, server_default="MANUAL"),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["report_id"], ["generated_reports.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("report_id", "version_number", name="uq_report_version"),
        )
        op.create_index("ix_report_versions_report_id", "report_versions", ["report_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_report_versions_report_id", table_name="report_versions")
    op.drop_table("report_versions")
    op.drop_index("ix_generated_reports_is_deleted", table_name="generated_reports")
    op.drop_index("ix_generated_reports_status", table_name="generated_reports")
    for column in ("is_deleted", "completed_at", "selected_document_ids", "source_types", "error", "status_text", "progress", "status"):
        op.drop_column("generated_reports", column)
