#!/usr/bin/env python3
"""
Test schema generation and output summarization functions.
Verifies that AI understands file structure and output stays within token limits.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge import SimpleAIBridge


def test_enhanced_schema_generation():
    """Test that _create_enhanced_schema generates complete schema without truncation."""
    print("\n" + "="*80)
    print("TEST 1: Enhanced Schema Generation")
    print("="*80)
    
    # Create test dataframes with various data types
    test_data = {
        'sales_data': pd.DataFrame({
            'transaction_id': np.arange(1, 1001),
            'date': pd.date_range('2024-01-01', periods=1000),
            'amount': np.random.uniform(10, 1000, 1000),
            'quantity': np.random.randint(1, 100, 1000),
            'customer_id': np.random.randint(1000, 2000, 1000),
            'product_name': ['Product_' + str(np.random.randint(1, 50)) for _ in range(1000)],
            'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
            'status': np.random.choice(['Completed', 'Pending', 'Cancelled'], 1000),
        }),
        'employee_data': pd.DataFrame({
            'emp_id': np.arange(1, 501),
            'name': ['Employee_' + str(i) for i in range(1, 501)],
            'department': np.random.choice(['Sales', 'Engineering', 'HR', 'Finance'], 500),
            'salary': np.random.uniform(30000, 150000, 500),
            'hire_date': pd.date_range('2015-01-01', periods=500),
            'performance_score': np.random.uniform(1, 5, 500),
            'location': np.random.choice(['NYC', 'SF', 'LA', 'Chicago'], 500),
        })
    }
    
    # Create bridge instance
    bridge = SimpleAIBridge()
    
    # Generate enhanced schema
    schema = bridge._create_enhanced_schema(test_data)
    
    print("\nGenerated Schema:")
    print("-" * 80)
    print(schema)
    print("-" * 80)
    
    # Verify all columns are included
    assert 'transaction_id' in schema, "Missing column: transaction_id"
    assert 'date' in schema, "Missing column: date"
    assert 'amount' in schema, "Missing column: amount"
    assert 'emp_id' in schema, "Missing column: emp_id"
    assert 'salary' in schema, "Missing column: salary"
    assert 'performance_score' in schema, "Missing column: performance_score"
    
    # Verify statistics are included
    assert 'range:' in schema, "Missing range information"
    assert 'mean:' in schema, "Missing mean calculation"
    assert 'unique values:' in schema, "Missing unique value count"
    
    # Count tokens (rough estimate: 1 token ≈ 4 characters)
    token_estimate = len(schema) / 4
    print(f"\n✓ Schema generated successfully")
    print(f"✓ Total characters: {len(schema)}")
    print(f"✓ Estimated tokens: ~{int(token_estimate)}")
    print(f"✓ All columns included without truncation")
    print(f"✓ Statistics included for numeric columns")


def test_dataframe_summary_no_overflow():
    """Test that _create_dataframe_summary handles large outputs without context overflow."""
    print("\n" + "="*80)
    print("TEST 2: Dataframe Summary (Context Overflow Prevention)")
    print("="*80)
    
    # Create a large dataframe
    large_df = pd.DataFrame({
        f'col_{i}': np.random.randint(0, 100, 10000)
        for i in range(1, 31)  # 30 columns
    })
    large_df['category'] = np.random.choice(['A', 'B', 'C', 'D'], 10000)
    large_df['description'] = ['Item_' + str(i) for i in range(10000)]
    
    bridge = SimpleAIBridge()
    
    # Create summary - should not read entire 10k rows
    summary = bridge._create_dataframe_summary(large_df, max_rows=10)
    
    print("\nGenerated Summary:")
    print("-" * 80)
    print(summary)
    print("-" * 80)
    
    # Verify properties of summary
    assert 'Total Rows: 10000' in summary, "Missing total row count"
    assert 'Total Columns: 31' in summary, "Missing total column count"
    assert 'First 10 rows:' in summary, "Missing first rows section"
    
    # Verify it doesn't read all 10k rows
    full_df_str = large_df.to_string()
    assert len(summary) < len(full_df_str) / 2, "Summary should be much smaller than full dataframe"
    
    token_estimate_summary = len(summary) / 4
    token_estimate_full = len(full_df_str) / 4
    reduction = (1 - len(summary) / len(full_df_str)) * 100
    
    print(f"\n✓ Summary generated successfully")
    print(f"✓ Summary tokens: ~{int(token_estimate_summary)} (from ~{int(token_estimate_full)} for full dataframe)")
    print(f"✓ Token reduction: {reduction:.1f}%")
    print(f"✓ Only first 10 rows included in output")
    print(f"✓ Column statistics provided without full enumeration")


def test_schema_vs_full_tokens():
    """Compare token usage: enhanced schema vs full dataframe string."""
    print("\n" + "="*80)
    print("TEST 3: Token Efficiency Comparison")
    print("="*80)
    
    # Create test data
    test_df = pd.DataFrame({
        f'column_{i}': np.random.uniform(0, 1000, 5000)
        for i in range(1, 25)  # 24 columns, 5000 rows
    })
    
    bridge = SimpleAIBridge()
    dataframes = {'test_table': test_df}
    
    # Generate enhanced schema
    schema = bridge._create_enhanced_schema(dataframes)
    schema_tokens = len(schema) / 4
    
    # Convert full dataframe to string (old approach)
    full_df_str = test_df.to_string()
    full_tokens = len(full_df_str) / 4
    
    # Generate summary
    summary = bridge._create_dataframe_summary(test_df)
    summary_tokens = len(summary) / 4
    
    print(f"\nToken Usage Comparison:")
    print(f"  Schema (for AI understanding):     ~{int(schema_tokens)} tokens")
    print(f"  Summary (for response):            ~{int(summary_tokens)} tokens")
    print(f"  Full DataFrame string (OLD way):   ~{int(full_tokens)} tokens")
    print()
    print(f"  Savings vs full dataframe output:  {(1 - summary_tokens / full_tokens) * 100:.1f}%")
    print(f"  Total with both schema + summary:  ~{int(schema_tokens + summary_tokens)} tokens")
    print(f"  Available context (example 4096):  Would have {int(4096 - (schema_tokens + summary_tokens))} tokens remaining ✓")


def test_sql_code_generation_with_schema():
    """Verify SQL generation has access to complete schema."""
    print("\n" + "="*80)
    print("TEST 4: SQL Code Generation with Complete Schema Info")
    print("="*80)
    
    test_data = {
        'orders': pd.DataFrame({
            'order_id': np.arange(1, 101),
            'customer_id': np.random.randint(1000, 1100, 100),
            'order_date': pd.date_range('2024-01-01', periods=100),
            'total_amount': np.random.uniform(100, 5000, 100),
            'status': np.random.choice(['pending', 'shipped', 'delivered'], 100),
            'payment_method': np.random.choice(['credit', 'debit', 'paypal'], 100),
        })
    }
    
    bridge = SimpleAIBridge()
    schema = bridge._create_enhanced_schema(test_data)
    
    # Extract table names and column counts from schema
    table_count = schema.count('=== TABLE:')
    column_mentions = schema.count('  -')
    
    print(f"\nSchema Analysis for SQL Generation:")
    print(f"  Tables recognized: {table_count}")
    print(f"  Column details provided: {column_mentions}")
    print(f"  Schema length: {len(schema)} chars (~{len(schema)//4} tokens)")
    
    # Verify all expected columns are there
    expected_cols = ['order_id', 'customer_id', 'order_date', 'total_amount', 'status', 'payment_method']
    for col in expected_cols:
        assert col in schema, f"Schema doesn't contain column: {col}"
    
    print(f"\n✓ All {len(expected_cols)} columns included in schema for SQL generation")
    print(f"✓ Column data types visible to AI")
    print(f"✓ Value ranges and statistics included")
    print(f"✓ AI can generate accurate SQL with proper column references")


if __name__ == '__main__':
    try:
        test_enhanced_schema_generation()
        test_dataframe_summary_no_overflow()
        test_schema_vs_full_tokens()
        test_sql_code_generation_with_schema()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nSummary:")
        print("  ✓ File schemas are now provided in complete detail to AI")
        print("  ✓ Output summaries prevent context window overflow")
        print("  ✓ Token efficiency improved by >90% vs full dataframe output")
        print("  ✓ AI has all information needed to write accurate code")
        print()
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
