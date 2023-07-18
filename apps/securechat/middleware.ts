import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
 
// This function can be marked `async` if using `await` inside
export function middleware(request: NextRequest) {
// //   return NextResponse.redirect(new URL('/home', request.url));
// const headersList = headers();
// const referer = headersList.get('X-Auth-Request-Email');
// console.log("hell header", referer)
const requestHeaders = new Headers(request.headers);
const userContext = requestHeaders.get('X-Auth-Request-Email') ?? "Anonymous";
if( userContext !== null){
  const response = NextResponse.next();
  response.cookies.set('X-Auth-Request-Email', userContext);
  return response;
}
}
 
// See "Matching Paths" below to learn more
export const config = {
  matcher: '/',
};