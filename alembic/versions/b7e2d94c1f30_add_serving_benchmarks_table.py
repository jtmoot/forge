"""add serving_benchmarks table

Revision ID: b7e2d94c1f30
Revises: a3f7c8e21b94
Create Date: 2026-10-29 13:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b7e2d94c1f30'
down_revision: Union[str, Sequence[str], None] = 'a3f7c8e21b94'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('serving_benchmarks',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('model_name', sa.String(length=255), nullable=False),
    sa.Column('model_version', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=20), server_default='running', nullable=False),
    sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('runtime', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('started_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_serving_benchmarks_model_name'), 'serving_benchmarks', ['model_name'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_serving_benchmarks_model_name'), table_name='serving_benchmarks')
    op.drop_table('serving_benchmarks')
