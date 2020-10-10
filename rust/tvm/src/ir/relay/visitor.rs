use super::Expr;

macro_rules! downcast_match {
    ($id:ident; { $($t:ty => $arm:expr $(,)? )+ , else => $default:expr }) => {
        $( if let Ok($id) = $id.downcast_clone::<$t>() { $arm } else )+
        { $default }
    }
}

trait ExprVisitorMut {
    fn visit(&mut self, expr: Expr) {
        downcast_match!(expr; {
            else => {
                panic!()
            }
        });
    }

    fn visit(&mut self, expr: Expr);
}

// trait ExprTransformer {
//     fn
// }
