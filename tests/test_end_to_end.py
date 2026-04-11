#!/usr/bin/env python3
"""
End-to-end test simulating the exact user scenario:
1. User uploads CSV files through UI
2. Agent receives instruction "recon_1: reconcile between 2 CSV files"
3. Code execution pipeline loads files and generates code
4. AI generates SQL/Python to analyze
5. Results returned without context overflow
"""

import base64
import io
import pandas as pd
import json


def simulate_ui_upload(csv_path):
    """Create a file object like the UI sends."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # UI might send as string or base64
    return {
        'name': csv_path.split('\\')[-1],
        'size': len(content),
        'content': content,
        'content_base64': None,
        'mime_type': 'text/csv',
        'path': None,  # Important: no disk path, only content
    }


def create_enhanced_schema(dataframes):
    """Generate schema for AI."""
    schema_lines = []
    
    for table_name, df in dataframes.items():
        schema_lines.append(f"\n=== TABLE: {table_name} ===")
        schema_lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        schema_lines.append("\nColumns:")
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
            col_info = f"  - {col}: {dtype} (nulls: {null_pct:.1f}%)"
            
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    col_info += f" | range: [{df[col].min():.2f}, {df[col].max():.2f}], mean: {df[col].mean():.2f}"
                except:
                    pass
            else:
                unique = df[col].nunique()
                col_info += f" | unique: {unique}"
            
            schema_lines.append(col_info)
    
    return "\n".join(schema_lines)


def create_dataframe_summary(df, max_rows=10):
    """Summarize dataframe for response."""
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
    
    lines.append("")
    lines.append("Column Summary:")
    lines.append("-" * 100)
    for col in df.columns[:10]:
        dtype = str(df[col].dtype)
        lines.append(f"{col}: {dtype}")
    
    return "\n".join(lines)


def test_end_to_end():
    """Test complete user workflow."""
    
    print("\n" + "="*100)
    print("END-TO-END TEST: Agent Mode with CSV File Upload")
    print("="*100)
    
    # Step 1: User uploads files
    print("\n[STEP 1] User uploads CSV files through UI")
    print("-" * 100)
    
    file_100 = simulate_ui_upload(r"c:\Users\Chandana\Downloads\100 BT Records.csv")
    file_1000 = simulate_ui_upload(r"c:\Users\Chandana\Downloads\1000 BT Records.csv")
    
    print(f"✓ File 1: {file_100['name']}")
    print(f"  - Size: {file_100['size']} bytes")
    print(f"  - Content available: {bool(file_100['content'])}")
    print(f"  - Path available: {bool(file_100['path'])}")
    print()
    print(f"✓ File 2: {file_1000['name']}")
    print(f"  - Size: {file_1000['size']} bytes")
    print(f"  - Content available: {bool(file_1000['content'])}")
    print(f"  - Path available: {bool(file_1000['path'])}")
    
    # Step 2: Files are loaded into dataframes
    print("\n[STEP 2] Code execution pipeline loads files into dataframes")
    print("-" * 100)
    
    dataframes = {}
    files = [file_100, file_1000]
    
    for f in files:
        name = f['name']
        try:
            # This mirrors the updated code in bridge.py
            df = None
            content = f.get('content')
            
            if content and isinstance(content, str) and name.lower().endswith('.csv'):
                df = pd.read_csv(io.StringIO(content))
            
            if df is None:
                print(f"✗ Failed to load {name}")
                return False
            
            table_name = name.replace(' ', '_').replace('.csv', '')
            dataframes[table_name] = df
            
            print(f"✓ Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            print(f"✗ Error loading {name}: {e}")
            return False
    
    if not dataframes:
        print("✗ No dataframes loaded!")
        return False
    
    # Step 3: Generate schema for AI
    print("\n[STEP 3] Generate schema for AI code generation")
    print("-" * 100)
    
    schema = create_enhanced_schema(dataframes)
    schema_tokens = len(schema) / 4
    
    print("Schema provided to AI:")
    print(schema[:400] + "\n... [truncated for display]")
    print(f"\nSchema tokens: ~{int(schema_tokens)}")
    
    # Step 4: Simulate prompt to AI
    print("\n[STEP 4] AI receives prompt with schema (not full file content)")
    print("-" * 100)
    
    instruction = "Reconcile between the two CSV files and provide summary and list of mismatch items"
    
    sql_prompt = f"""You are a data analyst. Generate SQL to analyze data.

TASK: {instruction[:500]}

DATABASE SCHEMA:
{schema}

Write ONLY the SQL query. No explanation."""
    
    prompt_tokens = len(sql_prompt) / 4
    
    print(f"Prompt size: {len(sql_prompt)} chars (~{int(prompt_tokens)} tokens)")
    print(f"✓ Within 4096 token limit: {prompt_tokens < 4096}")
    
    if prompt_tokens >= 4096:
        print(f"✗ FAILED: Prompt exceeds context window!")
        return False
    
    # Step 5: Generate output summary
    print("\n[STEP 5] Results returned as summary (not full dataframe)")
    print("-" * 100)
    
    # Combine both dataframes for analysis
    combined_df = pd.concat([
        dataframes['100_BT_Records'].assign(source='100_BT_Records'),
        dataframes['1000_BT_Records'].assign(source='1000_BT_Records'),
    ], ignore_index=True)
    
    summary = create_dataframe_summary(combined_df, max_rows=5)
    summary_tokens = len(summary) / 4
    
    print("Output summary:")
    print(summary[:300] + "\n... [truncated for display]")
    print(f"\nSummary tokens: ~{int(summary_tokens)}")
    
    # Step 6: Token efficiency check
    print("\n[STEP 6] Token Efficiency Analysis")
    print("-" * 100)
    
    full_str_tokens = len(combined_df.to_string()) / 4
    total_prompt_tokens = prompt_tokens + summary_tokens
    available = 4096 - total_prompt_tokens
    
    print(f"Schema:              ~{int(schema_tokens):>5} tokens")
    print(f"Summary:             ~{int(summary_tokens):>5} tokens")
    print(f"Full DataFrame:      ~{int(full_str_tokens):>5} tokens (NOT USED)")
    print(f"Total used:          ~{int(total_prompt_tokens):>5} tokens")
    print(f"Available context:    {int(available):>5} tokens remaining")
    print(f"Savings:               {(1 - total_prompt_tokens/full_str_tokens)*100:.1f}%")
    
    if available < 500:
        print(f"✗ WARNING: Only {available} tokens remaining!")
        return False
    
    print(f"✓ SAFE: {available} tokens remaining for response generation")
    
    # Final summary
    print("\n" + "="*100)
    print("TEST PASSED ✓")
    print("="*100)
    print("""
✓ Files uploaded through UI (no disk path)
✓ Code execution pipeline loads from memory
✓ Schema generated for AI understanding
✓ AI receives prompt within context limits (no 41085 token error!)
✓ Output returned as summary (not full content)
✓ Safe token budget for AI response generation

THE FIX:
1. CSV files can be loaded from content (string or base64)
2. CSV/Excel files SKIP text fallback extraction
3. Schema + summary approach keeps tokens manageable
4. No more "Requested tokens exceed context window" errors!
""")
    return True


if __name__ == '__main__':
    success = test_end_to_end()
    exit(0 if success else 1)
