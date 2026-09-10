"""Add persistent conversation summaries.

Revision ID: 0012_conversation_memory
Revises: 0011_chat_request_idempotency
"""

from alembic import op
import sqlalchemy as sa


revision = "0012_conversation_memory"
down_revision = "0011_chat_request_idempotency"
branch_labels = None
depends_on = None


TABLE_NAME = "conversation_summaries"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME in inspector.get_table_names():
        return
    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("active_entities", sa.JSON(), nullable=False),
        sa.Column("discussed_topics", sa.JSON(), nullable=False),
        sa.Column("user_preferences", sa.JSON(), nullable=False),
        sa.Column("pending_questions", sa.JSON(), nullable=False),
        sa.Column("covered_until_message_id", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("summary_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "session_id", name="uq_conversation_summary_session"),
    )
    op.create_index("ix_conversation_summaries_user_id", TABLE_NAME, ["user_id"], unique=False)
    op.create_index("ix_conversation_summaries_session_id", TABLE_NAME, ["session_id"], unique=True)
    op.create_index("ix_conversation_summaries_covered_until_message_id", TABLE_NAME, ["covered_until_message_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME in inspector.get_table_names():
        op.drop_table(TABLE_NAME)
