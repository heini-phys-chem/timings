
subroutine matrixmul(a, b, m, n, c)
  implicit none
  integer:: m, n
  real*8, dimension(m, n), intent(in):: a
  real*8, dimension(m, n), intent(in):: b
  real*8, dimension(m, n), intent(out):: c

  c = MATMUL(a, b)

end subroutine matrixmul
