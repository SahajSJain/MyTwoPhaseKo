#!/usr/bin/env python3
import os
import sys

def find_cpp_hpp_files(root_dir):
    """
    Recursively find all .cpp and .hpp files in the directory tree.
    
    Args:
        root_dir: The root directory to start searching from
        
    Returns:
        A sorted list of file paths
    """
    cpp_hpp_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.cpp', '.hpp')):
                full_path = os.path.join(dirpath, filename)
                cpp_hpp_files.append(full_path)
    
    return sorted(cpp_hpp_files)

def create_combined_file(files, output_filename='Code.txt'):
    """
    Combine all the content from the given files into a single output file.
    
    Args:
        files: List of file paths to combine
        output_filename: Name of the output file
    """
    try:
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            for file_path in files:
                # Write a separator with the file path
                output_file.write(f"\n{'='*80}\n")
                output_file.write(f"File: {file_path}\n")
                output_file.write(f"{'='*80}\n\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as input_file:
                        content = input_file.read()
                        output_file.write(content)
                        
                        # Ensure there's a newline at the end of each file's content
                        if not content.endswith('\n'):
                            output_file.write('\n')
                            
                except Exception as e:
                    error_msg = f"Error reading file {file_path}: {str(e)}\n"
                    print(error_msg, file=sys.stderr)
                    output_file.write(f"/* {error_msg} */\n")
        
        print(f"Successfully created {output_filename}")
        
    except Exception as e:
        print(f"Error creating output file: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    # Get the current directory or use command line argument
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = '.'
    
    # Verify the directory exists
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanning directory: {os.path.abspath(root_dir)}")
    
    # Find all .cpp and .hpp files
    files = find_cpp_hpp_files(root_dir)
    
    if not files:
        print("No .cpp or .hpp files found!")
        sys.exit(0)
    
    print(f"Found {len(files)} file(s):")
    for file in files:
        print(f"  - {file}")
    
    # Create the combined file
    create_combined_file(files)
    
    # Print summary
    output_size = os.path.getsize('Code.txt')
    print(f"\nCombined {len(files)} files into Code.txt ({output_size:,} bytes)")

if __name__ == "__main__":
    main()
