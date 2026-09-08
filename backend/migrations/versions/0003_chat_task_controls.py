"""Add task controls and clarification candidates to chat messages.

Revision ID: 0003_chat_task_controls
Revises: 0002_expand_long_text
"""

from alembic import op
import sqlalchemy as sa


revision = "0003_chat_task_controls"
down_revision = "0002_expand_long_text"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_messages", sa.Column("task_id", sa.String(length=64), nullable=True))
    op.create_index("ix_chat_messages_task_id", "chat_messages", ["task_id"], unique=False)
    op.add_column("chat_messages", sa.Column("clarification_candidates", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_messages", "clarification_candidates")
    op.drop_index("ix_chat_messages_task_id", table_name="chat_messages")
    op.drop_column("chat_messages", "task_id")
