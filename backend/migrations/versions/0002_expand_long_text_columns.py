"""Expand document and generated-content fields to MySQL LONGTEXT.

Revision ID: 0002_expand_long_text
Revises: 0001_initial
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


revision = "0002_expand_long_text"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {
        "documents": ("raw_text", "parsed_text", "summary"),
        "chat_messages": ("question", "answer"),
        "generated_reports": ("content_markdown",),
    }
    for table, names in columns.items():
        for name in names:
            op.alter_column(
                table,
                name,
                existing_type=sa.Text(),
                type_=mysql.LONGTEXT(),
                existing_nullable=name != "content_markdown",
            )


def downgrade() -> None:
    columns = {
        "documents": ("raw_text", "parsed_text", "summary"),
        "chat_messages": ("question", "answer"),
        "generated_reports": ("content_markdown",),
    }
    for table, names in columns.items():
        for name in names:
            op.alter_column(
                table,
                name,
                existing_type=mysql.LONGTEXT(),
                type_=sa.Text(),
                existing_nullable=name != "content_markdown",
            )
