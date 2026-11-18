program symmetric_group_permutations
    implicit none
    integer, parameter :: n = 12  ! Size of the symmetric group
    integer :: perm(n), c(n)
    integer :: i, temp

    ! Initialize the permutation array and control array
    do i = 1, n
        perm(i) = i
        c(i) = 0
    end do

    ! Print the initial permutation
    ! call print_permutation(perm, n)

    i = 1
    do while (i < n)
        if (c(i) < i) then
            ! Determine the index to swap with
            if (mod(i, 2) == 0) then
                temp = perm(1)
                perm(1) = perm(i + 1)
                perm(i + 1) = temp
            else
                temp = perm(c(i) + 1)
                perm(c(i) + 1) = perm(i + 1)
                perm(i + 1) = temp
            end if

            ! Print the current permutation
            ! call print_permutation(perm, n)

            ! Update control array
            c(i) = c(i) + 1
            i = 1
        else
            c(i) = 0
            i = i + 1
        end if
    end do

contains

    subroutine print_permutation(perm, size)
        integer :: size
        integer, intent(in) :: perm(size)
        integer :: k
        do k = 1, size
            write(*, '(I0)', advance='no') perm(k)
            if (k < size) write(*, '(A)', advance='no') ', '
        end do
        write(*, *)
    end subroutine print_permutation

end program symmetric_group_permutations
