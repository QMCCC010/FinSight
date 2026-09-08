"""Add explainable analysis metadata to chat messages.

Revision ID: 0009_chat_analysis_metadata
Revises: 0008_crawl_run_source_types
"""

from alembic import op
import sqlalchemy as sa


revision = "0009_chat_analysis_metadata"
down_revision = "0008_crawl_run_source_types"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {item["name"] for item in sa.inspect(bind).get_columns("chat_messages")}
    if "analysis_metadata" not in columns:
        op.add_column("chat_messages", sa.Column("analysis_metadata", sa.JSON(), nullable=True))
        bind.execute(sa.text("UPDATE chat_messages SET analysis_metadata = '{}' WHERE analysis_metadata IS NULL"))
        op.alter_column("chat_messages", "analysis_metadata", existing_type=sa.JSON(), nullable=False)


def downgrade() -> None:
    columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("chat_messages")}
    if "analysis_metadata" in columns:
        op.drop_column("chat_messages", "analysis_metadata")
