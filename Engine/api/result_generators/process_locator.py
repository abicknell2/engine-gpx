class ProcessLocator:
    """
    Tracks the origin and classification of each process in a manufacturing model.

    Each process is identified as one of the following:
    - Mainline process:    is_main = True,  feeder = None
    - Feeder process:      is_main = False, feeder = feeder_line_id
    - Secondary process:   is_main = False, feeder = None

    This structure supports logic for navigating feeder line dependencies.
    """

    def __init__(self, process_flow: list, feeder_processes: list, secondary_processes: list | None = None) -> None:
        """
        Constructs the process locator by mapping process IDs to their type and source.

        Args:
            process_flow (list): List of dictionaries representing main process steps.
            feeder_processes (list): List of feeder process dictionaries.
            secondary_processes (list | None): Optional list of secondary process dictionaries.
        """
        self._map: dict[str, tuple[bool, str | None]] = {}

        for p in process_flow:
            self._map[p["id"]] = (True, None)

        if secondary_processes:
            for p in secondary_processes:
                # Only set if not already marked as main
                self._map.setdefault(p["id"], (False, None))

        for p in feeder_processes:
            self._map[p["id"]] = (False, p["feederLine"])

    def __getitem__(self, process_id: str) -> tuple[bool, str | None]:
        """
        Retrieves the classification of the given process ID.

        Args:
            process_id (str): The identifier of the process.

        Returns:
            tuple: A tuple of (is_main, feeder_line_id or None).
        """
        return self._map[process_id]

    def is_main(self, process_id: str) -> bool:
        """
        Indicates whether a process is part of the main process flow.

        Args:
            process_id (str): The identifier of the process.

        Returns:
            bool: True if it is a mainline process, otherwise False.
        """
        return self._map.get(process_id, (False, None))[0]

    def downstream_feeder(self, process_id: str) -> str | None:
        """
        Provides the feeder line ID that feeds into this process, if applicable.

        Args:
            process_id (str): The identifier of the process.

        Returns:
            str | None: The ID of the upstream feeder line, or None if not applicable.
        """
        return self._map.get(process_id, (False, None))[1]

    def has(self, process_id: str) -> bool:
        """
        Checks whether the process ID is known to the locator.

        Args:
            process_id (str): The identifier of the process.

        Returns:
            bool: True if the process exists, otherwise False.
        """
        return process_id in self._map

    def is_secondary(self, process_id: str) -> bool:
        """
        Identifies whether a process is classified as secondary (not main, not feeder).

        Args:
            process_id (str): The identifier of the process.

        Returns:
            bool: True if the process is secondary, otherwise False.
        """
        return not self.is_main(process_id) and self.downstream_feeder(process_id) is None
