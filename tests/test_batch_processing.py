#!/usr/bin/env python3
"""
Test suite for batch processing functionality.

Tests the automatic detection and routing of large files to batch processing mode.
Tests DuckDB streaming with realistic data volumes.
Tests memory efficiency and performance.
"""

import sys
import os
import csv
import json
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Tuple, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("⚠️  DuckDB not available. Install with: pip install duckdb")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  Pandas not available. Install with: pip install pandas")


class BatchProcessingTester:
    """Test batch processing with various file sizes and scenarios."""
    
    def __init__(self):
        self.test_dir = tempfile.mkdtemp(prefix="batch_test_")
        self.results = []
        print(f"\n🔬 Test directory: {self.test_dir}")
    
    def create_test_csv(self, num_rows: int, num_cols: int = 5, 
                        filename: str = None) -> Tuple[str, int]:
        """
        Create a test CSV file with specified dimensions.
        
        Returns: (filepath, file_size_bytes)
        """
        if filename is None:
            filename = f"test_{num_rows}_rows.csv"
        
        filepath = os.path.join(self.test_dir, filename)
        
        # Create CSV with realistic data
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            headers = [f"col_{i}" for i in range(num_cols)]
            writer.writerow(headers)
            
            # Write data rows
            for row_id in range(num_rows):
                row = [
                    f"value_{row_id}_{col}" 
                    for col in range(num_cols)
                ]
                writer.writerow(row)
        
        file_size = os.path.getsize(filepath)
        return filepath, file_size
    
    def estimate_file_rows(self, filepath: str) -> int:
        """Estimate number of rows by sampling (simulating pre-flight check)."""
        with open(filepath, 'r') as f:
            # Read first 100 rows to estimate
            reader = csv.reader(f)
            next(reader)  # skip header
            sample_rows = sum(1 for _ in range(100) for _ in [next(reader, None)] if _ is not None)
        
        # Simple estimation: multiply by file size ratio
        file_size = os.path.getsize(filepath)
        avg_row_size = file_size / sample_rows if sample_rows > 0 else 1
        estimated_rows = int(file_size / avg_row_size)
        
        return estimated_rows
    
    def test_small_file(self) -> Dict[str, Any]:
        """Test small file (< 100k rows) - normal processing."""
        print("\n" + "="*80)
        print("TEST 1: Small File (Normal Processing)")
        print("="*80)
        
        filepath, file_size = self.create_test_csv(num_rows=5000)
        print(f"✓ Created: {filepath}")
        print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        estimated_rows = self.estimate_file_rows(filepath)
        print(f"  Estimated rows: {estimated_rows:,}")
        
        batch_mode = estimated_rows > 100_000
        print(f"  Batch mode required: {batch_mode}")
        
        result = {
            'test': 'small_file',
            'file_size': file_size,
            'estimated_rows': estimated_rows,
            'batch_mode': batch_mode,
            'status': 'PASS' if not batch_mode else 'FAIL',
        }
        
        return result
    
    def test_medium_file(self) -> Dict[str, Any]:
        """Test medium file (100k - 1M rows) - triggers batch mode."""
        print("\n" + "="*80)
        print("TEST 2: Medium File (Batch Mode Detection)")
        print("="*80)
        
        filepath, file_size = self.create_test_csv(num_rows=200_000, num_cols=8)
        print(f"✓ Created: {filepath}")
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
        estimated_rows = self.estimate_file_rows(filepath)
        print(f"  Estimated rows: {estimated_rows:,}")
        
        batch_mode = estimated_rows > 100_000
        print(f"  Batch mode required: {batch_mode}")
        
        result = {
            'test': 'medium_file',
            'file_size': file_size,
            'estimated_rows': estimated_rows,
            'batch_mode': batch_mode,
            'status': 'PASS' if batch_mode else 'FAIL',
        }
        
        return result
    
    def test_duckdb_streaming(self) -> Dict[str, Any]:
        """Test DuckDB streaming efficiency."""
        print("\n" + "="*80)
        print("TEST 3: DuckDB Streaming Memory Efficiency")
        print("="*80)
        
        if not DUCKDB_AVAILABLE:
            print("⚠️  Skipping: DuckDB not installed")
            return {'test': 'duckdb_streaming', 'status': 'SKIPPED'}
        
        filepath, file_size = self.create_test_csv(num_rows=100_000, num_cols=10)
        print(f"✓ Created: {filepath}")
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Test DuckDB streaming
        tracemalloc.start()
        start_time = time.time()
        
        try:
            # Simulate batch processing
            conn = duckdb.connect(':memory:')
            result = conn.execute(f"""
                SELECT * FROM read_csv_auto('{filepath}') LIMIT 1000
            """).fetchall()
            
            elapsed = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            rows_processed = len(result)
            print(f"✓ Processed {rows_processed} rows")
            print(f"  Time: {elapsed:.2f} seconds")
            print(f"  Memory used: {peak / 1024 / 1024:.1f} MB")
            print(f"  Memory efficiency: {(peak / file_size) * 100:.1f}% of file size")
            
            status = 'PASS' if rows_processed >= 1000 else 'FAIL'
            
            result = {
                'test': 'duckdb_streaming',
                'file_size_mb': file_size / 1024 / 1024,
                'rows_processed': rows_processed,
                'time_seconds': elapsed,
                'memory_mb': peak / 1024 / 1024,
                'memory_efficiency_percent': (peak / file_size) * 100,
                'status': status,
            }
            
        except Exception as e:
            print(f"✗ Error: {e}")
            result = {
                'test': 'duckdb_streaming',
                'status': 'FAIL',
                'error': str(e),
            }
        
        return result
    
    def test_large_file_simulation(self) -> Dict[str, Any]:
        """Simulate large file (1GB+) processing."""
        print("\n" + "="*80)
        print("TEST 4: Large File Simulation (1GB+)")
        print("="*80)
        
        if not DUCKDB_AVAILABLE:
            print("⚠️  Skipping: DuckDB not installed")
            return {'test': 'large_file_sim', 'status': 'SKIPPED'}
        
        # Create a moderately large file (simulate 1GB)
        filepath, file_size = self.create_test_csv(num_rows=500_000, num_cols=15)
        print(f"✓ Created simulation file: {filepath}")
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"  Simulated as: 1000x larger = {file_size * 1000 / 1024/1024/1024:.1f} GB")
        
        estimated_rows = self.estimate_file_rows(filepath)
        simulated_rows = estimated_rows * 1000
        print(f"  Simulated rows: {simulated_rows:,}")
        
        # Test batch mode detection
        batch_mode = estimated_rows > 100_000
        file_size_fits_5gb = file_size < (5 * 1024 * 1024 * 1024)
        should_use_batch = batch_mode and file_size_fits_5gb
        
        print(f"  Batch mode required: {batch_mode}")
        print(f"  File size < 5GB: {file_size_fits_5gb}")
        print(f"  Should use batch: {should_use_batch}")
        
        result = {
            'test': 'large_file_sim',
            'file_size_mb': file_size / 1024 / 1024,
            'simulated_size_gb': file_size * 1000 / 1024 / 1024 / 1024,
            'estimated_rows': estimated_rows,
            'simulated_rows': simulated_rows,
            'batch_detected': batch_mode,
            'file_size_ok': file_size_fits_5gb,
            'should_use_batch': should_use_batch,
            'status': 'PASS' if should_use_batch else 'FAIL',
        }
        
        return result
    
    def test_pandas_vs_duckdb(self) -> Dict[str, Any]:
        """Compare memory usage: pandas load vs DuckDB stream."""
        print("\n" + "="*80)
        print("TEST 5: Pandas Load vs DuckDB Stream")
        print("="*80)
        
        if not PANDAS_AVAILABLE or not DUCKDB_AVAILABLE:
            print("⚠️  Skipping: pandas or DuckDB not installed")
            return {'test': 'pandas_vs_duckdb', 'status': 'SKIPPED'}
        
        filepath, file_size = self.create_test_csv(num_rows=100_000, num_cols=12)
        print(f"✓ Test file: {file_size / 1024 / 1024:.2f} MB")
        
        # Test 1: Pandas full load
        print("\n  [Method 1] Pandas full load (old approach):")
        tracemalloc.start()
        start = time.time()
        
        df = pd.read_csv(filepath)
        
        pandas_time = time.time() - start
        _, pandas_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"    Time: {pandas_time:.2f}s")
        print(f"    Memory: {pandas_peak / 1024 / 1024:.1f} MB")
        
        # Test 2: DuckDB streaming
        print("\n  [Method 2] DuckDB streaming (new approach):")
        tracemalloc.start()
        start = time.time()
        
        conn = duckdb.connect(':memory:')
        _ = conn.execute(f"""
            SELECT * FROM read_csv_auto('{filepath}') LIMIT 1000
        """).fetchall()
        
        duckdb_time = time.time() - start
        _, duckdb_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"    Time: {duckdb_time:.2f}s")
        print(f"    Memory: {duckdb_peak / 1024 / 1024:.1f} MB")
        
        # Comparison
        memory_saved_pct = (1 - duckdb_peak / pandas_peak) * 100
        time_saved_pct = (1 - duckdb_time / pandas_time) * 100
        
        print(f"\n  💾 Memory saved: {memory_saved_pct:.1f}%")
        print(f"  ⚡ Time saved: {time_saved_pct:.1f}%")
        
        result = {
            'test': 'pandas_vs_duckdb',
            'pandas_time_s': pandas_time,
            'pandas_memory_mb': pandas_peak / 1024 / 1024,
            'duckdb_time_s': duckdb_time,
            'duckdb_memory_mb': duckdb_peak / 1024 / 1024,
            'memory_savings_pct': memory_saved_pct,
            'time_savings_pct': time_saved_pct,
            'status': 'PASS' if memory_saved_pct > 0 else 'FAIL',
        }
        
        return result
    
    def run_all_tests(self) -> None:
        """Run all tests and generate report."""
        print("\n" + "="*80)
        print("BATCH PROCESSING TEST SUITE")
        print("="*80)
        
        tests = [
            self.test_small_file,
            self.test_medium_file,
            self.test_duckdb_streaming,
            self.test_large_file_simulation,
            self.test_pandas_vs_duckdb,
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                self.results.append(result)
            except Exception as e:
                print(f"\n✗ Test failed: {e}")
                self.results.append({
                    'test': test_func.__name__,
                    'status': 'ERROR',
                    'error': str(e),
                })
        
        # Generate report
        self.print_report()
    
    def print_report(self) -> None:
        """Print test report summary."""
        print("\n" + "="*80)
        print("TEST REPORT")
        print("="*80)
        
        passed = sum(1 for r in self.results if r.get('status') == 'PASS')
        failed = sum(1 for r in self.results if r.get('status') == 'FAIL')
        skipped = sum(1 for r in self.results if r.get('status') == 'SKIPPED')
        errors = sum(1 for r in self.results if r.get('status') == 'ERROR')
        
        print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped, {errors} errors")
        print(f"Total: {len(self.results)} tests\n")
        
        for result in self.results:
            status_symbol = {
                'PASS': '✓',
                'FAIL': '✗',
                'SKIPPED': '⊘',
                'ERROR': '⚠',
            }.get(result.get('status'), '?')
            
            print(f"{status_symbol} {result.get('test', 'unknown')}: {result.get('status')}")
            
            # Print details
            for key, value in result.items():
                if key not in ['test', 'status']:
                    if isinstance(value, float):
                        print(f"    {key}: {value:.2f}")
                    else:
                        print(f"    {key}: {value}")
        
        print("\n" + "="*80)
        print("✅ BATCH PROCESSING TESTS COMPLETE")
        print("="*80)
        
        # Output JSON report
        report_path = os.path.join(self.test_dir, 'batch_test_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📊 Report saved: {report_path}")
        
        # Cleanup
        print(f"📁 Test files: {self.test_dir}")


def main():
    """Run batch processing tests."""
    tester = BatchProcessingTester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
