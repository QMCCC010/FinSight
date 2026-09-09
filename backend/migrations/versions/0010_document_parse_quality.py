"""Track document parser quality and summary provenance.

Revision ID: 0010_document_parse_quality
Revises: 0009_chat_analysis_metadata
"""

import json

from alembic import op
import sqlalchemy as sa


revision = "0010_document_parse_quality"
down_revision = "0009_chat_analysis_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {item["name"] for item in sa.inspect(bind).get_columns("documents")}
    if "parser_version" not in columns:
        op.add_column("documents", sa.Column("parser_version", sa.String(length=64), nullable=True))
    if "parse_quality" not in columns:
        op.add_column("documents", sa.Column("parse_quality", sa.Float(), nullable=True))
    if "parse_warnings" not in columns:
        op.add_column("documents", sa.Column("parse_warnings", sa.JSON(), nullable=True))
        bind.execute(
            sa.text("UPDATE documents SET parse_warnings = :warnings WHERE parse_warnings IS NULL"),
            {"warnings": json.dumps([], ensure_ascii=False)},
        )
        op.alter_column("documents", "parse_warnings", existing_type=sa.JSON(), nullable=False)
    if "summary_method" not in columns:
        op.add_column("documents", sa.Column("summary_method", sa.String(length=24), nullable=True))
        bind.execute(sa.text("UPDATE documents SET summary_method = 'UNKNOWN' WHERE summary_method IS NULL"))
        op.alter_column("documents", "summary_method", existing_type=sa.String(length=24), nullable=False)


def downgrade() -> None:
    columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("documents")}
    for name in ("summary_method", "parse_warnings", "parse_quality", "parser_version"):
        if name in columns:
            op.drop_column("documents", name)
