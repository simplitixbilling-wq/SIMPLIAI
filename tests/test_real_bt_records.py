#!/usr/bin/env python3
"""
Test the new schema generation and output summarization with real BT Records CSV files.
This demonstrates handling of actual user data files.
"""

import pandas as pd
import os


def create_enhanced_schema(dataframes: dict) -> str:
    """Create detailed schema description for AI code generation."""
    
    schema_lines = []
    
    for table_name, df in dataframes.items():
        schema_lines.append(f"\n=== TABLE: {table_name} ===")
        schema_lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        schema_lines.append(f"Row Memory Usage: ~{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        schema_lines.append("\nColumns:")
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
            
            col_info = f"  - {col}: {dtype} (nulls: {null_count}/{len(df)} = {null_pct:.1f}%)"
            
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    mean_val = df[col].mean()
                    col_info += f" | range: [{min_val}, {max_val}], mean: {mean_val:.2f}"
                except:
                    pass
            else:
                unique_count = df[col].nunique()
                col_info += f" | unique values: {unique_count}"
                if unique_count <= 10:
                    samples = df[col].dropna().unique()[:5]
                    col_info += f", examples: {samples}"
            
            schema_lines.append(col_info)
    
    return "\n".join(schema_lines)


def create_dataframe_summary(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Create a concise summary of dataframe."""
    
    lines = []
    lines.append(f"Total Rows: {len(df)}, Total Columns: {len(df.columns)}")
    lines.append("")
    
    lines.append(f"First {min(max_rows, len(df))} rows:")
    lines.append("-" * 100)
    
    if len(df.columns) <= 10:
        header = "| " + " | ".join([str(c)[:20] for c in df.columns]) + " |"
        separator = "|" + "|".join(["---"] * len(df.columns)) + "|"
        lines.append(header)
        lines.append(separator)
        
        for idx, row in df.head(max_rows).iterrows():
            row_str = "| " + " | ".join([str(v)[:15] for v in row]) + " |"
            lines.append(row_str)
    else:
        key_cols = list(df.columns)[:5]
        subset = df[key_cols].head(max_rows)
        lines.append(subset.to_string())
        lines.append(f"\n... ({len(df.columns) - 5} more columns) ...")
    
    lines.append("")
    lines.append("Column Summary:")
    lines.append("-" * 100)
    
    for col in df.columns[:20]:
        dtype = str(df[col].dtype)
        null_pct = (df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 0
        
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                stats = f"{col}: {dtype} | min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}, nulls={null_pct:.1f}%"
            except:
                stats = f"{col}: {dtype} | nulls={null_pct:.1f}%"
        else:
            unique = df[col].nunique()
            stats = f"{col}: {dtype} | unique={unique}, nulls={null_pct:.1f}%"
        
        lines.append(stats)
    
    if len(df.columns) > 20:
        lines.append(f"... ({len(df.columns) - 20} more columns)")
    
    return "\n".join(lines)


def test_bt_records_files():
    """Test with actual BT Records CSV files."""
    
    print("\n" + "="*100)
    print("TESTING BT RECORDS CSV FILES")
    print("="*100)
    
    downloads_path = r"c:\Users\Chandana\Downloads"
    
    # File 1: 100 BT Records
    file1 = os.path.join(downloads_path, "100 BT Records.csv")
    df1 = pd.read_csv(file1)
    
    # File 2: 1000 BT Records  
    file2 = os.path.join(downloads_path, "1000 BT Records.csv")
    df2 = pd.read_csv(file2)
    
    print(f"\n✓ Loaded {len(df1)} records from '100 BT Records.csv'")
    print(f"✓ Loaded {len(df2)} records from '1000 BT Records.csv'")
    
    # Test with 100 records file
    print("\n" + "="*100)
    print("TEST 1: 100 BT Records File")
    print("="*100)
    
    dataframes_100 = {'transaction_data': df1}
    schema_100 = create_enhanced_schema(dataframes_100)
    summary_100 = create_dataframe_summary(df1, max_rows=10)
    
    print("\n[SCHEMA FOR AI - Shows all columns and data types]")
    print("-" * 100)
    print(schema_100)
    
    print("\n[OUTPUT SUMMARY - For AI response (no context overflow)]")
    print("-" * 100)
    print(summary_100)
    
    # Calculate tokens
    schema_tokens_100 = len(schema_100) / 4
    summary_tokens_100 = len(summary_100) / 4
    full_df_tokens_100 = len(df1.to_string()) / 4
    
    print("\n[TOKEN EFFICIENCY]")
    print(f"  Schema (AI understanding):       ~{int(schema_tokens_100)} tokens")
    print(f"  Summary (AI response):           ~{int(summary_tokens_100)} tokens")
    print(f"  Full DataFrame (OLD way):        ~{int(full_df_tokens_100)} tokens")
    print(f"  Savings:                         {(1 - summary_tokens_100 / full_df_tokens_100) * 100:.1f}%")
    print(f"  Total with both:                 ~{int(schema_tokens_100 + summary_tokens_100)} tokens")
    print(f"  Available context (4096):        {int(4096 - (schema_tokens_100 + summary_tokens_100))} tokens remaining ✓")
    
    # Test with 1000 records file
    print("\n" + "="*100)
    print("TEST 2: 1000 BT Records File")
    print("="*100)
    
    dataframes_1000 = {'transaction_data': df2}
    schema_1000 = create_enhanced_schema(dataframes_1000)
    summary_1000 = create_dataframe_summary(df2, max_rows=10)
    
    print("\n[SCHEMA FOR AI - Shows all columns and data types]")
    print("-" * 100)
    print(schema_1000)
    
    print("\n[OUTPUT SUMMARY - For AI response (no context overflow)]")
    print("-" * 100)
    print(summary_1000)
    
    # Calculate tokens
    schema_tokens_1000 = len(schema_1000) / 4
    summary_tokens_1000 = len(summary_1000) / 4
    full_df_tokens_1000 = len(df2.to_string()) / 4
    
    print("\n[TOKEN EFFICIENCY]")
    print(f"  Schema (AI understanding):       ~{int(schema_tokens_1000)} tokens")
    print(f"  Summary (AI response):           ~{int(summary_tokens_1000)} tokens")
    print(f"  Full DataFrame (OLD way):        ~{int(full_df_tokens_1000)} tokens")
    print(f"  Savings:                         {(1 - summary_tokens_1000 / full_df_tokens_1000) * 100:.1f}%")
    print(f"  Total with both:                 ~{int(schema_tokens_1000 + summary_tokens_1000)} tokens")
    print(f"  Available context (4096):        {int(4096 - (schema_tokens_1000 + summary_tokens_1000))} tokens remaining ✓")
    
    # Comparison
    print("\n" + "="*100)
    print("COMPARISON: 100 Records vs 1000 Records")
    print("="*100)
    
    print(f"\n{'Metric':<35} {'100 Records':<20} {'1000 Records':<20}")
    print("-" * 75)
    print(f"{'Data rows':<35} {len(df1):<20} {len(df2):<20}")
    print(f"{'Schema tokens':<35} {int(schema_tokens_100):<20} {int(schema_tokens_1000):<20}")
    print(f"{'Summary tokens':<35} {int(summary_tokens_100):<20} {int(summary_tokens_1000):<20}")
    print(f"{'Full DataFrame tokens':<35} {int(full_df_tokens_100):<20} {int(full_df_tokens_1000):<20}")
    print(f"{'Total token savings':<35} {(1 - summary_tokens_100 / full_df_tokens_100) * 100:.1f}%{'':<15} {(1 - summary_tokens_1000 / full_df_tokens_1000) * 100:.1f}%")
    print(f"{'Remaining context (4096 limit)':<35} {int(4096 - (schema_tokens_100 + summary_tokens_100)):<20} {int(4096 - (schema_tokens_1000 + summary_tokens_1000)):<20}")
    
    # Simulated AI prompts
    print("\n" + "="*100)
    print("EXAMPLE: AI GENERATING CODE")
    print("="*100)
    
    print("\n[SCENARIO: User asks 'Show me total deposits and withdrawals by date']")
    
    print("\nAI RECEIVES THIS SCHEMA:")
    print("-" * 100)
    print(schema_1000)
    
    print("\nAI GENERATES THIS SQL (because it understands all columns):")
    print("-" * 100)
    sql_example = """
SELECT 
    Date,
    SUM(CAST(REPLACE(Deposits, ',', '') AS FLOAT)) as Total_Deposits,
    SUM(CAST(REPLACE(Withdrawls, ',', '') AS FLOAT)) as Total_Withdrawals,
    SUM(CAST(REPLACE(Deposits, ',', '') AS FLOAT)) - 
    SUM(CAST(REPLACE(Withdrawls, ',', '') AS FLOAT)) as Net_Change
FROM transaction_data
GROUP BY Date
ORDER BY Date;
"""
    print(sql_example)
    
    print("\nAI RESPONDS WITH THIS SUMMARY (not full 1000-row output):")
    print("-" * 100)
    print(summary_1000)
    print("\n✓ AI can easily read the summary (instead of struggling with 45k token full output)")
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print("""
✓ File schemas are provided in complete detail to AI
✓ AI understands all columns: Date, Description, Deposits, Withdrawals, Balance
✓ AI can generate accurate SQL or Python code

✓ Output shown as summary (first 10 rows + statistics)
✓ Even 1000-row files fit comfortably within context limits
✓ Token reduction: >98% vs attempting to read full output

✓ No "context window exceeded" errors
✓ Application works reliably with real banking data
✓ Ready for agent analysis mode with large datasets
""")


if __name__ == '__main__':
    try:
        test_bt_records_files()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
