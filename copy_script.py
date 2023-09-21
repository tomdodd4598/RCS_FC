import shutil


def main():
    for i in range(26, 51, 2):
        # Source file path
        src_dir = f'C:\\Users\\Duan\\OneDrive - University of Edinburgh\\Year 4\\毕设\\doi_10.5061_dryad.k6t1rj8__v11\\n{i}_m14\\'

        # Target file path
        dest_dir = f'Full Circuit\\e0_{i}\\'

        for j in range(0, 10):
            try:
                src_file = f'{src_dir}measurements_n{i}_m14_s{j}_e0_pEFGH.txt'
                dest_file = f'{dest_dir}measurements_n{i}_m14_s{j}_e0_pEFGH.txt'
                # Copy file contents and permission bits
                shutil.copy(src_file, dest_file)
                # Also tries to maintain the metadata of the file (such as modification time, etc.)
                # shutil.copy2(src_file, dest_file)
            except shutil.SameFileError as _:
                src_file = f'{src_dir}measurements_n{i}_m14_s1{j}_e0_pEFGH.txt'
                dest_file = f'{dest_dir}measurements_n{i}_m14_s{j}_e0_pEFGH.txt'
                # Copy file contents and permission bits
                shutil.copy(src_file, dest_file)
                # Also tries to maintain the metadata of the file (such as modification time, etc.)
                # shutil.copy2(src_file, dest_file)


if __name__ == '__main__':
    main()
