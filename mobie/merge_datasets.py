
def inputs_are_valid(mobie_projects, datasets):
    # TODO: This function should check for the following:
    #  - The input mobie projects must exist
    #  - The input mobie projects are valid
    #  - The datasets must exist in the respective mobie projects
    #  - There must be no datasets of equal name in both inputs
    print('Warning: Validity check not implemented, always returning True.')
    return True


def make_empty_output_dataset(out_folder):

    import os
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)


def parse_sources_in_dataset(dataset):
    """
    returns a dictionary with the format:

    {
        "source_name1": [
            "bdv.n5",
            "image"
        ],
        "source_name2": [
            "bdv.n5",
            "segmentation"
        ]
    }
    """

    from mobie.metadata.dataset_metadata import read_dataset_metadata
    metadata = read_dataset_metadata(dataset)

    if 'sources' not in metadata:
        return []

    def detect_data_type(source_info):
        if 'bdv.n5' in source_info['image']['imageData']:
            return 'bdv.n5'
        elif 'bdv.hdf5' in source_info['image']['imageData']:
            return 'bdv.hdf5'
        else:
            return None

    def detect_source_type(source_name, metadata):

        if 'imageDisplay' in metadata['views'][source_name]['sourceDisplays'][0]:
            return 'image'
        elif 'segmentationDisplay' in metadata['views'][source_name]['sourceDisplays'][0]:
            return 'segmentation'
        else:
            return None

    return {
        source: [detect_data_type(val), detect_source_type(source, metadata)]
        for source, val in metadata['sources'].items()
    }


def populate_mobie_dataset(
        target_root_folder, target_dataset_name,
        input_datasets, input_sources,
        copy=False,
        verbose=False
):

    from mobie.utils import require_dataset_and_view
    from mobie.metadata.source_metadata import add_source_to_dataset
    from mobie.xml_utils import copy_xml_with_newpath
    from mobie.xml_utils import get_data_path
    import os

    sources = {dataset: parse_sources_in_dataset(dataset) for dataset in input_datasets}

    tasks = [
        {'dataset': ds, 'source': src, 'info': sources[ds][src]}
        for idx, ds in enumerate(input_datasets) for src in input_sources[idx]
    ]
    target_dataset_folder = os.path.join(target_root_folder, target_dataset_name)

    if verbose:
        print(f'tasks = {tasks}')

    for task in tasks:
        if verbose:
            print(f'task = {task}')

        data_format = task['info'][0]
        source_type = task['info'][1]

        if verbose:
            print(f'data_format = {data_format}')

        view = require_dataset_and_view(
            root=target_root_folder,
            dataset_name=target_dataset_name,
            file_format=data_format,
            source_type=source_type,
            source_name=task['source'],
            menu_name=None,
            view=None,
            is_default_dataset=False,
            contrast_limits=[0, 255] if source_type == 'image' else None
        )

        input_base_name = os.path.join(task['dataset'], 'images', data_format.replace('.', '-'), f'{task["source"]}')
        input_xml_path = f'{input_base_name}.xml'
        input_data_path = get_data_path(input_xml_path, return_absolute_path=True)

        xml_path = os.path.join(target_dataset_folder, 'images', '{}', f'{task["source"]}.xml')
        xml_path = xml_path.format(data_format.replace('.', '-'))

        if copy:
            print('Copying data ...')
            from shutil import copytree
            target_data_path = f'{os.path.splitext(xml_path)[0]}.{"n5" if data_format == "bdv.n5" else "h5"}'
            if verbose:
                print(f'copytree( "{input_data_path}", "{target_data_path}" )')
            copytree(input_data_path, target_data_path)
            copy_xml_with_newpath(
                input_xml_path, xml_path, os.path.split(target_data_path)[1],
                path_type='relative', data_format=data_format
            )
        else:
            copy_xml_with_newpath(
                input_xml_path, xml_path, input_data_path,
                path_type='absolute', data_format=data_format
            )

        if verbose:
            print(f'target_dataset_folder = {target_dataset_folder}')
            print(f'task["source"] = {task["source"]}')
            print(f'xml_path = {xml_path}')
            print(f'view = {view}')
        add_source_to_dataset(
            target_dataset_folder, 'image', task["source"], xml_path,
            overwrite=True, view=view
        )


def merge_mobie_datasets(
        mobie_a,
        mobie_b,
        out_folder,
        out_dataset,
        sources_a=None,
        sources_b=None,
        copy_data=False,
        verbose=False
):

    if verbose:
        print(f'mobie_a = {mobie_a}')
        print(f'mobie_b = {mobie_b}')
        print(f'out_folder = {out_folder}')
        print(f'out_dataset = {out_dataset}')
        print(f'sources_a = {sources_a}')
        print(f'sources_b = {sources_b}')
        print(f'copy_data = {copy_data}')

    if not inputs_are_valid([mobie_a, mobie_b], [sources_a, sources_b]):
        print(f'Error: Non-valid inputs!')
        return

    populate_mobie_dataset(
        out_folder, out_dataset,
        [mobie_a, mobie_b], [sources_a, sources_b],
        copy=copy_data,
        verbose=verbose
    )


def main():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge two MoBIE datasets',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('mobie_a', type=str,
                        help='Location of first mobie dataset')
    parser.add_argument('mobie_b', type=str,
                        help='Location of second mobie dataset')
    parser.add_argument('out_folder', type=str,
                        help='Output folder where the results will be written to')
    parser.add_argument('out_dataset', type=str,
                        help='Name of the output dataset')
    parser.add_argument('--sources_a', type=str, nargs='+', default=None,
                        help='Which sources to use from the first mobie project, by default all images are used')
    parser.add_argument('--sources_b', type=str, nargs='+', default=None,
                        help='Which sources to use from the second mobie project, by default all images are used')
    parser.add_argument('-copy', '--copy_data', action='store_true',
                        help='Copy the data, by default only xml files are created in the output dataset')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    mobie_a = args.mobie_a
    mobie_b = args.mobie_b
    out_folder = args.out_folder
    out_dataset = args.out_dataset
    sources_a = args.sources_a
    sources_b = args.sources_b
    copy_data = args.copy_data
    verbose = args.verbose

    merge_mobie_datasets(
        mobie_a,
        mobie_b,
        out_folder,
        out_dataset,
        sources_a=sources_a,
        sources_b=sources_b,
        copy_data=copy_data,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
