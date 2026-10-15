program symmetric_group_permutations
    implicit none
    integer, parameter :: n = 11  ! Size of the symmetric group
    integer :: perm(n)
    logical :: used(n)
    integer :: i

    ! Initialize the arrays
    perm = 0
    used = .false.

    call generate_permutations(perm, used, 1, n)

contains

    recursive subroutine generate_permutations(perm, used, position, size)
        ! Declare types explicitly
        integer, intent(inout) :: perm(size)
        logical, intent(inout) :: used(size)
        integer, intent(in) :: position, size
        integer :: i

        if (position > size) then
            ! call print_permutation(perm, size)
        else
            do i = 1, size
                if (.not. used(i)) then
                    perm(position) = i
                    used(i) = .true.
                    call generate_permutations(perm, used, position + 1, size)
                    used(i) = .false.
                end if
            end do
        end if
    end subroutine generate_permutations

    subroutine print_permutation(perm, size)
        ! Declare types explicitly
        integer :: size
        integer, intent(in) :: perm(size)
        integer :: i
        do i = 1, size
            write(*, '(I0)', advance='no') perm(i)
            if (i < size) write(*, '(A)', advance='no') ', '
        end do
        write(*, *)
    end subroutine print_permutation

end program symmetric_group_permutations
