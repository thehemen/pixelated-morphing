from misc.misc import get_equal_values

dependencies = {
    'Eye': ['Head'],
    'Nose': ['Eye'],
    'Mouth': ['Nose', 'Eye'],
    'Hair': ['Head'],
    'Helmet': ['Head'],
    'Ear': ['Head'],
    'Eyebrow': ['Eye']
}

def find_dependencies_by_region(region_1, region_2):
    """Find dependencies between two non-paired image regions (head, nose, mouth, hair, helmet).

    Keyword arguments:
    region_1 - the follower image region
    region_2 - the leader image region
    """
    dependent_values = region_1.bbox.get_values()
    class_name = region_2.class_name

    if not region_2.is_paired:
        base_values = region_2.bbox.get_values()
        values = get_equal_values(dependent_values, base_values)

        if len(values) > 0:
            region_1.dependencies[class_name] = values
    else:
        base_values_1 = region_2.first.bbox.get_values()
        base_values_2 = region_2.second.bbox.get_values()

        values = get_equal_values(dependent_values, base_values_1)
        opposite_values = get_equal_values(dependent_values, base_values_2)

        # Mark regions as dependent if they have any equal coordinates.
        if len(values) > 0:
            region_1.dependencies[f'{class_name}_1'] = values
            region_1.dependencies[f'{class_name}_2'] = opposite_values

def find_dependencies_by_pair(region_1, region_2):
    """Find dependencies between two paired image regions (eye, ear, eyebrow).

    Keyword arguments:
    region_1 - the follower image region
    region_2 - the leader image region
    """
    dependent_values_1 = region_1.first.bbox.get_values()
    dependent_values_2 = region_1.second.bbox.get_values()
    class_name = region_2.class_name

    if not region_2.is_paired:
        base_values = region_2.bbox.get_values()
        values = get_equal_values(dependent_values_1, base_values)
        opposite_values = get_equal_values(dependent_values_2, base_values)

        if len(values) > 0:
            region_1.first.dependencies[class_name] = values
            region_1.second.dependencies[class_name] = opposite_values
    else:
        base_values_1 = region_2.first.bbox.get_values()
        base_values_2 = region_2.second.bbox.get_values()

        values = get_equal_values(dependent_values_1, base_values_1)
        opposite_values = get_equal_values(dependent_values_2, base_values_2)

        # Mark regions as dependent if they have any equal coordinates.
        if len(values) > 0:
            region_1.first.dependencies[f'{class_name}_1'] = values
            region_1.second.dependencies[f'{class_name}_2'] = opposite_values

            region_1.first.dependencies[f'{class_name}_1'] = values
            region_1.second.dependencies[f'{class_name}_2'] = opposite_values

def find_dependencies(labeled_im):
    """Find dependencies for all image regions."""
    for dependent_class, base_classes in dependencies.items():
        if dependent_class not in labeled_im.regions.keys():
            continue

        for base_class in base_classes:
            if base_class not in labeled_im.regions.keys():
                continue

            if not labeled_im.regions[dependent_class].is_paired:
                find_dependencies_by_region(labeled_im.regions[dependent_class],
                                            labeled_im.regions[base_class])
            else:
                find_dependencies_by_pair(labeled_im.regions[dependent_class],
                                          labeled_im.regions[base_class])
